"""Face->Age datasets"""
import re
import csv
import math
import logging
import importlib
import multiprocessing
from itertools import repeat
from datetime import date, timedelta
import torch
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm
from scipy.io import loadmat
from torchvision import transforms
from .base import _OnlineDataset, _RawDataset, _Source
from ..utils import save_obj, load_obj, file_exists, multiprocess_num_jobs


log = logging.getLogger(__name__)

face_recognition = None
cv2 = None


def init_face_rec_module():
    """
    Dynamically import face_recognition so that it won't waste gpu memory when
    preprocess is not required.
    """
    global cv2
    global face_recognition
    cv2 = cv2 or importlib.import_module('cv2')
    face_recognition = face_recognition or importlib.import_module('face_recognition')


class _FaceAgeDataset(_RawDataset, _OnlineDataset):
    """Base class of all face->age datasets"""

    def __init__(self, cfg, mode, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs, mode=mode)

        # force 101 classes (see __init__ of `FLSplit`)
        self.classes = set(range(101))
        self.input_size = 224
        self.cache_path = f"{self.root}/{self.__class__.__name__}.pkl"
        self.download_and_extrat()
        self.preprocess_and_cache_meta()
        self.dataset = load_obj(self.cache_path)

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.RandomCrop(self.input_size, padding=56),
                transforms.RandomRotation(5.000000000),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

    def skip_download_extract(self):
        return not self.dataset_updated()

    def re_preprocess(self):
        return not file_exists(self.meta_cache_path)

    def load_sample_descriptors(self):
        return list(range(len(self.dataset)))

    def get_data(self, desc):
        # convert grayscale to rgb
        image = self.dataset[desc][0]
        if image.mode == 'L':
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image)
            image = rgb_image
        return self.transform(desc[0])

    def get_label(self, desc):
        return self.dataset[desc][1]


class Wiki(_FaceAgeDataset):
    """IMDB-Wiki dataset (Wiki part)"""
    _root = 'data/src/face_age.wiki'

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, srcs=_Source(
                url='https://data.vision.ee.ethz.ch/'
                'cvl/rrothe/imdb-wiki/static/wiki_crop.tar',
                file='wiki_crop.tar',
                md5='f536eb7f5eae229ae8f286184364b42b',
                compress_type='tar'))

    def preprocess(self):
        init_face_rec_module()
        wiki_data = loadmat(self.root.joinpath('wiki_crop/wiki.mat')
                            .as_posix())['wiki'][0][0]
        wiki = list(filter(
            lambda x: 0 <= x[1] <= 100,
            [('wiki_crop/' + path[0],
              int((date.fromisoformat(f"{wiki_data[1][0][i]}-07-01") -
                   (date.fromordinal(int(wiki_data[0][0][i])) -
                    timedelta(days=366))).days / 365.25))
             for i, path in enumerate(wiki_data[2][0])
             if (not math.isinf(wiki_data[6][0][i]))
             and (wiki_data[0][0][i] > 366)
             and math.isnan(wiki_data[7][0][i])]))

        dataset = []
        with multiprocessing.Pool(multiprocess_num_jobs()) as pool:
            dataset = list(tqdm(pool.imap(
                _wiki_process_file, zip(repeat(self.root), wiki)),
                total=len(wiki)))
        dataset = list(filter(lambda x: x is not None, dataset))
        save_obj(dataset, self.cache_path)


def _wiki_process_file(args):
    root, data = args
    if len(data[0].strip()) == 0:
        return None

    aligned_face = align_and_crop(root.joinpath(data[0]).as_posix())

    if aligned_face is not None:
        image = Image.fromarray(aligned_face)
        image = image.resize((224, 224))
        return (image, data[1])
    return None


class AllAgeFaces(_FaceAgeDataset):
    """All-Age-Faces dataset"""
    _root = 'data/src/face_age.allagefaces'
    file_pattern = re.compile(r'\d{5}A(\d{2})\.jpg (0|1)')

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, srcs=_Source(
                url='https://www.dropbox.com/'
                's/a0lj1ddd54ns8qy/All-Age-Faces Dataset.zip',
                file='All-Age-Faces Dataset.zip',
                md5='c02186d20f46f234e777e8e80edb3267',
                compress_type='zip'))

    def preprocess(self):
        init_face_rec_module()
        # read list of images files
        files = []
        with open(self.root.joinpath(
                'All-Age-Faces Dataset/image sets/train.txt')
                .as_posix(), encoding='utf-8') as file:
            files = file.read().split('\n')
        with open(self.root.joinpath(
                'All-Age-Faces Dataset/image sets/val.txt')
                .as_posix(), encoding='utf-8') as file:
            files += file.read().split('\n')

        # preprocess images
        with multiprocessing.Pool(multiprocess_num_jobs()) as pool:
            dataset = list(tqdm(pool.imap(
                _allagefaces_process_file, zip(repeat(self.root), files)),
                total=len(files)))
            dataset = list(filter(lambda x: x is not None, dataset))
        save_obj(dataset, self.cache_path)


# unfortunately, multiprocessing can only pickle top-level functions
def _allagefaces_process_file(args):
    root, file = args
    if len(file.strip()) == 0:
        return None

    match_result = AllAgeFaces.file_pattern.match(file)
    assert match_result is not None
    label = int(match_result.groups()[0])

    aligned_face = align_and_crop(
        root.joinpath(
            f"All-Age-Faces Dataset/original images/{file.split(' ')[0]}"
        ).as_posix())

    if aligned_face is not None:
        image = Image.fromarray(aligned_face)
        image = image.resize((224, 224))
        return (image, label)
    return None


class AppaRealAge(_FaceAgeDataset):
    """APPA-Real-Age dataset"""
    _root = 'data/src/face_age.appa'
    sizes = {'train': 4113, 'test': 1978, 'valid': 1500}

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, srcs=_Source(
                url='http://158.109.8.102/AppaRealAge/appa-real-release.zip',
                file='appa-real-release.zip',
                md5='2701c6ec291b242e6b4f8a0a56087a43',
                compress_type='zip'))

    def preprocess(self):
        init_face_rec_module()
        dataset = []
        for orig_mode, size in AppaRealAge.sizes.items():
            with open(self.root.joinpath(
                f"appa-real-release/gt_avg_{orig_mode}.csv").as_posix(),
                encoding='utf-8') as file,\
                    multiprocessing.Pool(multiprocess_num_jobs()) as pool:
                reader = csv.DictReader(file)
                dataset += list(tqdm(pool.imap(
                    _appa_process_file,
                    zip(repeat(self.root), repeat(orig_mode), reader)),
                    total=size))
        dataset = list(filter(lambda x: x is not None, dataset))
        save_obj(dataset, self.cache_path)


def _appa_process_file(args):
    root, orig_mode, row = args
    aligned_face = align_and_crop(root.joinpath(
        f"appa-real-release/{orig_mode}/{row['file_name']}").as_posix())
    if aligned_face is not None:
        image = Image.fromarray(aligned_face)
        image = image.resize((224, 224))
        return (image, int(row['real_age']))
    return None


class UTKFaces(_FaceAgeDataset):
    """UTK-faces dataset"""
    _root = 'data/src/face_age.utk'
    file_pattern = re.compile(r'(\d+)_\d_\d_\d+\.jpg\.chip\.jpg')

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, srcs=_Source(
                url=None,
                file='UTKFace.tar.gz',
                md5='ae1a16905fbd795db921ff1d940df9cc',
                compress_type='tgz'))

    def preprocess(self):
        init_face_rec_module()
        files = list(self.root.joinpath('UTKFace').glob('*.jpg'))
        with multiprocessing.Pool(multiprocess_num_jobs()) as pool:
            dataset = list(tqdm(pool.imap(_utk_process_file, files),
                                total=len(files)))
        dataset = list(filter(lambda x: x is not None, dataset))
        save_obj(dataset, self.cache_path)


def _utk_process_file(file):
    aligned_face = align_and_crop(file.as_posix())
    if aligned_face is not None:
        match = UTKFaces.file_pattern.match(file.name)
        if match is not None:
            label = int(match.groups()[0])
            if 0 <= label <= 100:
                image = Image.fromarray(aligned_face)
                image = image.resize((224, 224))
                return (image, label)
    return None


def align_and_crop(filename):
    """Align faces by eyes and crop faces"""
    image = face_recognition.load_image_file(filename)
    locations = face_recognition.face_locations(image, model="hog")
    landmarks = face_recognition.face_landmarks(image)

    if len(locations) == 0 or len(landmarks) == 0:
        return None

    left_eye_pts = landmarks[0]['left_eye']
    right_eye_pts = landmarks[0]['right_eye']

    left_eye_center = np.array(left_eye_pts).mean(axis=0).astype("int")
    right_eye_center = np.array(right_eye_pts).mean(axis=0).astype("int")
    left_eye_center = (left_eye_center[0], left_eye_center[1])
    right_eye_center = (right_eye_center[0], right_eye_center[1])

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    desired_left_eye = (0.39, 0.5)
    desired_right_eye_x = 1 - desired_left_eye[0]
    desired_image_width = 224
    desired_image_height = 224

    dist = np.sqrt((dx**2) + (dy**2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0])
    desired_dist *= desired_image_width
    scale = desired_dist / dist

    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                   (left_eye_center[1] + right_eye_center[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tx = desired_image_width * 0.5
    ty = desired_image_height * desired_left_eye[1]
    M[0, 2] += (tx - eyes_center[0])
    M[1, 2] += (ty - eyes_center[1])

    (y2, x2, y1, x1) = locations[0]
    return cv2.warpAffine(
        image, M, (desired_image_width, desired_image_height),
        flags=cv2.INTER_CUBIC)


# accuracy metrics for age prediction specifically
def mae(output: torch.Tensor, ground_truth) -> float:
    preds = torch.sum(
        nn.Softmax(1)(output) * torch.arange(101).to(output.device), 1)
    return nn.L1Loss()(preds, ground_truth)


def acc(output: torch.Tensor, ground_truth) -> float:
    preds = torch.sum(
        nn.Softmax(1)(output) * torch.arange(101).to(output.device), 1)
    return (ground_truth == preds.round()).float().sum().item() / \
        len(ground_truth)


def one_off_acc(output: torch.Tensor, ground_truth) -> float:
    preds = torch.sum(
        nn.Softmax(1)(output) * torch.arange(101).to(output.device), 1)
    return ((ground_truth == preds.round()).float().sum().item() +
            (ground_truth + 1 == preds.round()).float().sum().item() +
            (ground_truth - 1 == preds.round()).float().sum().item()) / \
        len(ground_truth)
