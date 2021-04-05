import os
import re
import csv
import cv2
import PIL
import math
import torch
import random
import numpy as np
import multiprocessing
import face_recognition
from torch import nn
from tqdm import tqdm
from scipy.io import loadmat
from itertools import repeat
from torchvision import transforms
from datetime import date, timedelta
from oarf.datasets.base import OnlineDataset, Source
from oarf.datasets.fl_dataset import RawDataset

_NUM_JOBS = max(int(os.getenv("NUM_JOBS", multiprocessing.cpu_count() - 1)), 1)


class FaceAgeDataset(RawDataset):
    def __init__(self, mode, rotation_degree=5, *args, **kwargs):
        # force 101 classes (see __init__ of `FLDataset`)
        self.classes = set(range(101))
        self.input_size = 224
        super().__init__(*args, **kwargs)
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

    def get_data(self, sample):
        # convert grayscale to rgb
        image = sample[0]
        if image.mode == 'L':
            rgbImage = PIL.Image.new("RGB", image.size)
            rgbImage.paste(image)
            image = rgbImage
        return self.transform(sample[0])

    def get_label(self, sample):
        return sample[1]


class IMDB(FaceAgeDataset, OnlineDataset):
    pass


class Wiki(FaceAgeDataset, OnlineDataset):
    def __init__(self, mode='train', rotation_degree=5, *_, **__):
        self.cache_base = 'wiki'
        self.shuffle_seed = 0xdeadbeef
        super().__init__(
            mode=mode, rotation_degree=rotation_degree,
            root='data/src/face_age.wiki',
            srcs=Source(
                url='https://data.vision.ee.ethz.ch/'
                'cvl/rrothe/imdb-wiki/static/wiki_crop.tar',
                file='wiki_crop.tar',
                md5='f536eb7f5eae229ae8f286184364b42b',
                compress_type='tar'),
            cache_file=self.cache_base + '_{}.pkl'.format(self.shuffle_seed))

    def preprocess(self):
        wiki_data = loadmat(self.root.joinpath('wiki_crop/wiki.mat')
                            .as_posix())['wiki'][0][0]
        wiki = list(filter(
            lambda x: 0 <= x[1] <= 100,
            [('wiki_crop/' + path[0],
              int((date.fromisoformat('{}-07-01'.format(wiki_data[1][0][i])) -
                   (date.fromordinal(int(wiki_data[0][0][i])) -
                    timedelta(days=366))).days / 365.25))
             for i, path in enumerate(wiki_data[2][0])
             if (not math.isinf(wiki_data[6][0][i]))
             and (wiki_data[0][0][i] > 366)
             and math.isnan(wiki_data[7][0][i])]))

        dataset = []
        with multiprocessing.Pool(_NUM_JOBS) as pool:
            dataset = list(tqdm(pool.imap(
                _wiki_process_file, zip(repeat(self.root), wiki)),
                total=len(wiki)))
        dataset = list(filter(lambda x: x is not None, dataset))

        random.Random(self.shuffle_seed).shuffle(dataset)
        return {'train': dataset[:len(dataset) * 5 // 6],
                'test': dataset[len(dataset) * 5 // 6: len(dataset) * 11//12],
                'val': dataset[len(dataset) * 11//12:]}

    def load_train_dataset(self, _):
        return self.load_cache()['train']

    def load_eval_dataset(self, seed):
        return self.load_cache()['test']

    def load_test_dataset(self, seed):
        return self.load_cache()['val']


def _wiki_process_file(args):
    root, data = args
    if len(data[0].strip()) == 0:
        return

    aligned_face = align_and_crop(root.joinpath(data[0]).as_posix())

    if aligned_face is not None:
        image = PIL.Image.fromarray(aligned_face)
        image = image.resize((224, 224))
        return (image, data[1])
    else:
        return None


class AllAgeFaces(FaceAgeDataset, OnlineDataset):
    file_pattern = re.compile(r'\d{5}A(\d{2})\.jpg (0|1)')

    def __init__(self, mode='train', rotation_degree=5, *_, **__):
        self.cache_base = 'allagefaces'
        self.shuffle_seed = 0xdeadbeef
        super().__init__(
            mode=mode, rotation_degree=rotation_degree,
            root='data/src/face_age.allagefaces',
            srcs=Source(
                url='https://www.dropbox.com/'
                's/a0lj1ddd54ns8qy/All-Age-Faces Dataset.zip',
                file='All-Age-Faces Dataset.zip',
                md5='c02186d20f46f234e777e8e80edb3267',
                compress_type='zip'),
            cache_file=self.cache_base + '_{}.pkl'.format(self.shuffle_seed))

    def preprocess(self):
        # read list of images files
        files = []
        with open(self.root.joinpath(
                'All-Age-Faces Dataset/image sets/train.txt').as_posix()) as f:
            files = f.read().split('\n')
        with open(self.root.joinpath(
                'All-Age-Faces Dataset/image sets/val.txt').as_posix()) as f:
            files += f.read().split('\n')

        # preprocess images
        with multiprocessing.Pool(_NUM_JOBS) as pool:
            dataset = list(tqdm(pool.imap(
                _allagefaces_process_file, zip(repeat(self.root), files)),
                total=len(files)))
            dataset = list(filter(lambda x: x is not None, dataset))

        random.Random(self.shuffle_seed).shuffle(dataset)
        return {'train': dataset[:len(dataset) * 5 // 6],
                'test': dataset[len(dataset) * 5 // 6: len(dataset) * 11//12],
                'val': dataset[len(dataset) * 11//12:]}

    def load_train_dataset(self, _):
        return self.load_cache()['train']

    def load_eval_dataset(self, seed):
        return self.load_cache()['test']

    def load_test_dataset(self, seed):
        return self.load_cache()['val']


# unfortunately, multiprocessing can only pickle top-level functions
def _allagefaces_process_file(args):
    root, f = args
    if len(f.strip()) == 0:
        return

    label = int(AllAgeFaces.file_pattern.match(f).groups()[0])
    aligned_face = align_and_crop(
        root.joinpath('All-Age-Faces Dataset/original images/{}'
                      .format(f.split(' ')[0])).as_posix())

    if aligned_face is not None:
        image = PIL.Image.fromarray(aligned_face)
        image = image.resize((224, 224))
        return (image, label)
    else:
        return None


class AppaRealAge(FaceAgeDataset, OnlineDataset):
    def __init__(self, mode='train', rotation_degree=5, *_, **__):
        self.cache_base = 'appa'
        self.shuffle_seed = 0xdeadbeef
        self.sizes = {'train': 4113, 'test': 1978, 'valid': 1500}
        super().__init__(
            mode=mode, rotation_degree=rotation_degree,
            root='data/src/face_age.appa',
            srcs=Source(
                url='http://158.109.8.102/AppaRealAge/appa-real-release.zip',
                file='appa-real-release.zip',
                md5='2701c6ec291b242e6b4f8a0a56087a43',
                compress_type='zip'),
            cache_file=self.cache_base + '_{}.pkl'.format(self.shuffle_seed))

    def preprocess(self):
        dataset = []
        for orig_mode, size in self.sizes.items():
            with open(self.root.joinpath('appa-real-release/gt_avg_{}.csv'
                                         .format(orig_mode)).as_posix()) as f,\
                    multiprocessing.Pool(_NUM_JOBS) as pool:
                reader = csv.DictReader(f)
                dataset += list(tqdm(pool.imap(
                    _appa_process_file,
                    zip(repeat(self.root), repeat(orig_mode), reader)),
                    total=size))

        dataset = list(filter(lambda x: x is not None, dataset))

        random.Random(self.shuffle_seed).shuffle(dataset)
        return {'train': dataset[:len(dataset) * 5 // 6],
                'test': dataset[len(dataset) * 5 // 6: len(dataset) * 11//12],
                'val': dataset[len(dataset) * 11//12:]}

    def load_train_dataset(self, _):
        return self.load_cache()['train']

    def load_eval_dataset(self, seed):
        return self.load_cache()['test']

    def load_test_dataset(self, seed):
        return self.load_cache()['val']


def _appa_process_file(args):
    root, orig_mode, row = args
    aligned_face = align_and_crop(root.joinpath(
        'appa-real-release/{}/{}'
        .format(orig_mode, row['file_name'])).as_posix())
    if aligned_face is not None:
        image = PIL.Image.fromarray(aligned_face)
        image = image.resize((224, 224))
        return (image, int(row['real_age']))
    else:
        return None

# class OUIAudience(FaceAgeDataset, OnlineDataset):
#     def __init__(self, *_, **__):
#         self.cache_base = 'wiki'
#         self.shuffle_seed = 0xdeadbeef
#         super().__init__(
#             root='data/src/face_age.ouiaudience',
#             srcs=Source(
#                 url=None,
#                 file='faces.tar.gz',
#                 md5='5c9fd94d64ec7f6168fd5df2117c25c6',
#                 compress_type='tgz'),
#             cache_file=self.cache_base + '_{}.pkl'.format(self.shuffle_seed))
#
#     def preprocess(self):
#         pass


class UTKFaces(FaceAgeDataset, OnlineDataset):
    file_pattern = re.compile(r'(\d+)_\d_\d_\d+\.jpg\.chip\.jpg')

    def __init__(self, mode='train', rotation_degree=5, *_, **__):
        self.cache_base = 'utk'
        self.shuffle_seed = 0xdeadbeef
        super().__init__(
            mode=mode, rotation_degree=rotation_degree,
            root='data/src/face_age.utk',
            srcs=Source(
                url=None,
                file='UTKFace.tar.gz',
                md5='ae1a16905fbd795db921ff1d940df9cc',
                compress_type='tgz'),
            cache_file=self.cache_base + '_{}.pkl'.format(self.shuffle_seed))

    def preprocess(self):
        files = list(self.root.joinpath('UTKFace').glob('*.jpg'))
        with multiprocessing.Pool(_NUM_JOBS) as pool:
            dataset = list(tqdm(pool.imap(_utk_process_file, files),
                                total=len(files)))
        dataset = list(filter(lambda x: x is not None, dataset))

        random.Random(self.shuffle_seed).shuffle(dataset)
        return {'train': dataset[:len(dataset) * 5 // 6],
                'test': dataset[len(dataset) * 5 // 6: len(dataset) * 11//12],
                'val': dataset[len(dataset) * 11//12:]}

    def load_train_dataset(self, _):
        return self.load_cache()['train']

    def load_eval_dataset(self, seed):
        return self.load_cache()['test']

    def load_test_dataset(self, seed):
        return self.load_cache()['val']


def _utk_process_file(f):
    aligned_face = align_and_crop(f.as_posix())
    if aligned_face is not None:
        match = UTKFaces.file_pattern.match(f.name)
        if match is not None:
            label = int(match.groups()[0])
            if 0 <= label <= 100:
                image = PIL.Image.fromarray(aligned_face)
                image = image.resize((224, 224))
                return (image, label)
    return None


def align_and_crop(filename):
    """align faces by eyes and crop faces"""
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
def mae(output, ground_truth) -> float:
    preds = torch.sum(nn.Softmax(1)(output) * torch.arange(101).cuda(), 1)
    return nn.L1Loss()(preds, ground_truth)


def acc(output, ground_truth) -> float:
    preds = torch.sum(nn.Softmax(1)(output) * torch.arange(101).cuda(), 1)
    return (ground_truth == preds.round()).float().sum().item() / \
        len(ground_truth)


def one_off_acc(output, ground_truth) -> float:
    preds = torch.sum(nn.Softmax(1)(output) * torch.arange(101).cuda(), 1)
    return ((ground_truth == preds.round()).float().sum().item() +
            (ground_truth + 1 == preds.round()).float().sum().item() +
            (ground_truth - 1 == preds.round()).float().sum().item()) /\
        len(ground_truth)
