"""
Rewrite of the FEMNIST dataset
https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
"""
import glob
import random
import hashlib
import logging
import threading
import multiprocessing
from typing import Any, List, Tuple
from tqdm import tqdm
from torchvision import transforms, io
from .base import _RawDataset, _OnlineDataset, _Source
from ..utils import save_obj, load_obj, file_exists, multiprocess_num_jobs


log = logging.getLogger(__name__)


# This dataset uses lazy-loading.
class FEMNIST(_RawDataset, _OnlineDataset):
    """
    Federated SD19 dataset, group by author. Processing method first proposed
    by [Leaf](https://github.com/TalwalkarLab/leaf). This class uses pure
    python to preprocess the dataset
    """

    _root = 'data/src/femnist'
    num_writers = 3500
    num_images_by_writer = 814255
    num_images_by_class = 1545923

    # Use shared descs to reduce memory usage & speedup loading
    load_lock = threading.Lock()
    full_ds: List[List[str]] = []
    eval_ds: List[Tuple[Any, int]] = []
    test_ds: List[Tuple[Any, int]] = []

    def __init__(self, cfg, mode, split_id, *args, rotation_degree=5, **kwargs):

        assert cfg.data.fl_split.num <= FEMNIST.num_writers, \
            "Number of clients must be less or equal to the number of writers"
        assert split_id is None or 0 <= split_id < cfg.data.fl_split.num

        super().__init__(cfg, *args, **kwargs, srcs=[
            _Source(
                url='https://s3.amazonaws.com/nist-srd/SD19/by_class.zip',
                file='by_class.zip',
                md5='79572b1694a8506f2b722c7be54130c4',
                compress_type='zip'),
            _Source(
                url='https://s3.amazonaws.com/nist-srd/SD19/by_write.zip',
                file='by_write.zip',
                md5='a29f21babf83db0bb28a2f77b2b456cb',
                compress_type='zip')
        ])

        self.classes = list(range(62))
        self.split_id = split_id
        self.num_splits = cfg.data.fl_split.num
        self.cache_path = f"{self.root}/{self.__class__.__name__}.pkl"
        self.rotation_degree = rotation_degree
        self.download_and_extrat()
        self.preprocess_and_cache_meta()

        # Initialize shared descs
        with FEMNIST.load_lock:
            if not FEMNIST.full_ds:
                FEMNIST.full_ds = load_obj(self.cache_path)
                for desc in FEMNIST.full_ds:
                    random.Random(self.seed).shuffle(desc)

        # Initialize train dataset
        train, val, _ = self.train_eval_test
        if mode == 'train':
            self.dataset = FEMNIST.full_ds[self.split_id][
                :int(len(FEMNIST.full_ds[self.split_id])
                     * train / sum(self.train_eval_test))]

        # Initialize shared eval dataset
        if mode == 'eval':
            with FEMNIST.load_lock:
                if not FEMNIST.eval_ds:
                    FEMNIST.eval_ds = [
                        d for ds in self.full_ds[:self.num_splits]
                        for d in ds[
                            int(len(ds) * train / sum(self.train_eval_test)):
                            int(len(ds) * (train + val) / sum(self.train_eval_test))]]
            self.dataset = FEMNIST.eval_ds

        # Initialize shared test dataset
        if mode == 'test':
            with FEMNIST.load_lock:
                if not FEMNIST.test_ds:
                    FEMNIST.test_ds = [
                        d for ds in self.full_ds[:self.num_splits]
                        for d in ds[
                            int(len(ds) * (train + val) / sum(self.train_eval_test)):]]
            self.dataset = FEMNIST.test_ds

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomRotation(rotation_degree),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.transform = transforms.Compose([])

    def metadata(self):
        return {**super().metadata(), 'rotation_degree': self.rotation_degree}

    def skip_download_extract(self):
        return file_exists(self.meta_cache_path)

    def re_preprocess(self):
        return not file_exists(self.meta_cache_path)

    def preprocess(self):
        # The structure of the two subdirs are
        # by_class/<classes>/<folders containing images>/images
        # by_write/<folders containing writers>/<writer>/<types of images>/images
        # 1. load metadata and match hashes
        files_by_class = glob.glob(
            self.root.as_posix() + '/by_class/*/*/*.png')
        with multiprocessing.Pool(multiprocess_num_jobs()) as pool:
            # hash_to_class_and_path = {hash: (class_name, path), ...}
            hash_to_class_and_path = dict(tqdm(pool.imap(
                _hash_file_by_class, files_by_class, chunksize=1000),
                total=self.num_images_by_class))

        files_by_writer = glob.glob(
            self.root.as_posix() + '/by_write/*/*/*/*.png')
        with multiprocessing.Pool(multiprocess_num_jobs()) as pool:
            # hash_to_writer = {hash: writer_name, ...}
            hash_to_writer = dict(tqdm(pool.imap(
                _hash_file_by_writer, files_by_writer, chunksize=1000),
                total=self.num_images_by_writer))

        # hash_to_meta = {hash: (writer, class, path), ...}
        hash_to_meta = {}
        for key, value in tqdm(hash_to_class_and_path.items()):
            if key in hash_to_writer:
                hash_to_meta[key] = (hash_to_writer[key], *value)

        # 2. group by writer (and convert class_name and writer_name to int)
        writer_name_to_int = dict(zip(sorted(
            set(meta[0] for meta in hash_to_meta.values())), range(99999)))
        class_name_to_int = dict(zip(sorted(
            set(meta[1] for meta in hash_to_meta.values())), range(99999)))
        log.debug("Total %s classes", len(class_name_to_int))

        # writers_data[<writer_id>] = [(path, class),...]
        writers_data = [[] for _ in range(len(writer_name_to_int))]
        for wtr, cls, path in hash_to_meta.values():
            writers_data[writer_name_to_int[wtr]].append(
                (path, class_name_to_int[cls]))
        # writers_data = [l for l in writers_data if len(l) != 0]
        writers_data = [[
            (io.read_image(d[0]), d[1]) for d in l
        ] for l in tqdm(writers_data) if len(l) != 0]
        log.debug("Total %s valid writers", len(writers_data))

        # cache the dataset
        save_obj(writers_data, self.cache_path)

    def load_sample_descriptors(self):
        pass

    def load_train_descs(self):
        return list(range(len(self.dataset)))

    def load_eval_descs(self):
        return list(range(len(self.eval_ds)))

    def load_test_descs(self):
        return list(range(len(self.test_ds)))

    def get_data(self, desc):
        return self.transform(1. - (self.dataset[desc][0] / 255.))

    def get_label(self, desc):
        return self.dataset[desc][1]


def _hash_file_by_class(path):
    with open(path, 'rb') as file:
        chash = hashlib.md5(file.read()).hexdigest()
    class_name = path.split('/')[-3]
    return (chash, (class_name, path))


def _hash_file_by_writer(path):
    with open(path, 'rb') as file:
        chash = hashlib.md5(file.read()).hexdigest()
    writer_name = path.split('/')[-3]
    return (chash, writer_name)
