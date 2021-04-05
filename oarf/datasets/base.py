import re
import gzip
import shutil
import pickle
import pathlib
import logging
import tarfile
import zipfile
from typing import Iterable, Union
from torch.utils.data import Dataset
from oarf.datasets.utils import download_file, file_already_cached

log = logging.getLogger(__name__)


class RawDataset(Dataset):
    def load_train_dataset(self, seed):
        """Subclass should implement this method. Must be deterministic."""
        raise NotImplementedError

    def load_eval_dataset(self, seed):
        """Subclass should implement this method. Must be deterministic."""
        raise NotImplementedError

    def load_test_dataset(self, seed):
        """Subclass should implement this method. Must be deterministic."""
        raise NotImplementedError

    def get_data(self, sample):
        """Subclass should implement this method and return data"""
        raise NotImplementedError

    def get_label(self, sample):
        """Subclass should implement this method and return label"""
        raise NotImplementedError


class Source:
    def __init__(self, url: str, file: str, md5: str, compress_type=None):
        """Compress type: zip | tar | tgz | rar """
        self.url = url
        self.file = file
        self.md5 = md5
        self.compress_type = compress_type


class OnlineDataset:
    def __init__(self, root: str, srcs: Union[Source, Iterable[Source]],
                 cache_file: str, cache_md5=None, *args, **kwargs):
        super().__init__()

        self.root = pathlib.Path(root)
        self.cache_path = self.root.joinpath(cache_file).as_posix()

        if self.check_cache(cache_md5):
            log.info("{}: Cache exists, skip preprocessing."
                     .format(self.__class__.__name__))
            return

        if isinstance(srcs, Iterable):
            for src in srcs:
                self.download(src)
                self.extract(src)
        else:
            self.download(srcs)
            self.extract(srcs)

        log.info("{}: Preprocessing ...".format(self.__class__.__name__))
        self.cache(self.preprocess())

    def check_cache(self, md5):
        if file_already_cached(self.cache_path, md5):
            return True
        return False

    def download(self, src: Source):
        self.root.mkdir(parents=True, exist_ok=True)
        src_path = self.root.joinpath(src.file).as_posix()
        if src.url is None:
            log.warning("No url provided")
            if not self.root.joinpath(src.file).is_file():
                log.error("'{}' not found, please download '{}' "
                          "manually and place it under '{}', "
                          " and make sure its md5 is {}."
                          .format(src_path, src.file,
                                  self.root.as_posix(), src.md5))
                raise Exception("Source file not found")
        else:
            download_file(src.url, src.md5, src_path)

    def extract(self, src: Source):
        log.info("Extracting {} ...".format(src.file))
        src_path = self.root.joinpath(src.file).as_posix()
        if src.compress_type == 'zip':
            with zipfile.ZipFile(src_path, 'r') as zip_ref:
                zip_ref.extractall(self.root.as_posix())
        elif src.compress_type == 'tar':
            with tarfile.open(src_path, 'r') as f:
                f.extractall(self.root.as_posix())
        elif src.compress_type == 'tgz':
            with tarfile.open(src_path, 'r:gz') as f:
                f.extractall(self.root.as_posix())
        elif src.compress_type == 'gz':
            out_file = re.sub('.gz$', '', src.file)
            out_file = (out_file if out_file != src.file
                        else src.file + '.unzipped')
            out_path = self.root.joinpath(out_file).as_posix()
            with gzip.open(src_path, 'rb') as fi, open(out_path, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
        else:
            raise Exception("Unknown compression type")

    def preprocess(self):
        """Subclass should implement this"""
        raise NotImplementedError

    def cache(self, obj):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(obj, f)

    def load_cache(self):
        with open(self.cache_path, 'rb') as f:
            return pickle.load(f)
