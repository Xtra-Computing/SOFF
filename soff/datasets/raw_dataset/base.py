"""Base of all raw datasets"""
import os
import re
import gzip
import shutil
import random
import pathlib
import logging
import tarfile
import zipfile
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable, Union, List, Any, Dict, Optional
from ..utils import download_file, metadata_updated, save_dict
from ...utils.arg_parser import ArgParseOption, options, require

log = logging.getLogger(__name__)


@require(
    "federation.seed",
    "data.raw.fold",
    "data.raw.train_eval_test")
@options(
    ArgParseOption(
        'f', 'fold', type=int, default=0, metavar='FOLD',
        help="Which fold to use (for N-fold cross validation)"),
    ArgParseOption(
        'tet', 'train-eval-test', default=[10, 1, 1],
        type=int, nargs=3, metavar='RATIO',
        help="Tran-eval-test split ratio."))
class _CachedDataset:
    """
    Hadles metadata caching.
    Data pre-processing and caching fuctionalities are handled in subclasses.
    """
    _root = f"data/src/{__name__}"

    def __init__(self, cfg, *args, **kwargs):
        assert len(cfg.data.raw.train_eval_test) == 3

        if hasattr(super(), '__dict__'):
            super().__init__(*args, **kwargs)

        self.root = pathlib.Path(self.__class__._root)
        pathlib.Path(self.root).mkdir(parents=True, exist_ok=True)

        # Some common metadatas affecting the dataset loading process
        self.seed = cfg.federation.seed
        self.fold = cfg.data.raw.fold % sum(cfg.data.raw.train_eval_test)
        self.train_eval_test = cfg.data.raw.train_eval_test

        cname = self.__class__.__name__
        self.meta: Dict[str, Any] = {}
        self.meta_cache_path = self.root.joinpath(f"{cname}_meta.pkl")

    def preprocess_and_cache_meta(self):
        """Preprocess the dataset and cache the metdata"""
        cname = self.__class__.__name__
        log.info("%s: seed: %s, cache path: %s", cname, self.seed, self.meta_cache_path)

        self.meta = self.metadata()

        if self.re_preprocess():
            log.info("%s: Preprocessing ...", cname)
            self.preprocess()
        else:
            log.info("%s: Cache valid, skip preprocessing ...", cname)

        if metadata_updated(self.meta, self.meta_cache_path):
            save_dict(self.meta, self.meta_cache_path)

    def metadata(self) -> Dict[str, Any]:
        """
        Return a dict of metadatas to be cached.
        This method can be overriden if subclass has additional metadatas. e.g.
        `return {**super().metadata(), 'key': <value>, ...}`
        """
        return {
            'seed': self.seed,
            'fold': self.fold,
            'train_eval_test': self.train_eval_test
        }

    def re_preprocess(self) -> bool:
        """
        Whether (re)preprocess is needed. Most of the time testing the
        existence of the metadata file is sufficient. However, if the
        preprocessing requires randomized operations or is depends on extra
        variables, additional tests on those variables are needed.
        """
        return True

    def preprocess(self) -> None:
        """Pre-processing. Must be deterministic."""

    def dataset_updated(self) -> bool:
        """
        Returning `True` will force the higher level FLSplit subclass
        instances to evict their cache and re-compose the federated dataset.
        """
        return metadata_updated(self.meta, self.meta_cache_path)


class _RawDataset(_CachedDataset, ABC):
    """
    Base class of all raw datasets. Handles dataset shuffling & splitting.
    """

    @abstractmethod
    def load_sample_descriptors(self) -> List[Any]:
        """
        Subclass should implement this method. Must be deterministic.
        Note: Don't directly return samples content here (such as images),
        otherwise a lot of memory will be wasted. If need to load the dataset
        directly from memory, use a member to store the actual dataset and
        override get_data to return the actual sample content.
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self, desc) -> Any:
        """Subclass should implement this method and return data"""
        raise NotImplementedError

    @abstractmethod
    def get_label(self, desc) -> int:
        """Subclass should implement this method and return label"""
        raise NotImplementedError

    def __load_folds(self, start, num):
        desc = self.load_sample_descriptors()
        random.Random(self.seed).shuffle(desc)

        total = sum(self.train_eval_test)
        fold_size = len(desc) // sum(self.train_eval_test)

        return desc[start * fold_size: (start + num) * fold_size] \
            if start + num <= total \
            else desc[:((start + num) % total) * fold_size] + \
            desc[start * fold_size:]

    def load_train_descs(self) -> List[Any]:
        """Load the training dataset descriptor list"""
        train, _, _ = self.train_eval_test
        return self.__load_folds(self.fold, train)

    def load_eval_descs(self) -> List[Any]:
        """Load the evaluation dataset descriptor list"""
        train, val, _ = self.train_eval_test
        return self.__load_folds(self.fold + train, val)

    def load_test_descs(self) -> List[Any]:
        """Load the test dataset descriptor list"""
        train, val, test = self.train_eval_test
        return self.__load_folds(self.fold + train + val, test)


@dataclass
class _Source:
    """Defines a source for download"""
    url: Optional[str]
    """Surce URL"""
    file: str
    """Filename to save the downloaded content as"""
    md5: str
    """MD5 for the downloaded file"""
    compress_type: str
    """Compress type: zip | tar | tgz | rar"""


class _OnlineDataset(ABC):
    """This class automatically download content from online sources"""
    _root = f"data/src/{__name__}"

    def __init__(self, srcs: Union[_Source, Iterable[_Source]], *args, **kwargs):
        if hasattr(super(), '__dict__'):
            super().__init__(*args, **kwargs)

        self.srcs = srcs
        self.download_path = pathlib.Path(self.__class__._root)

    def download_and_extrat(self):
        """Download and extract dataset from source."""
        cname = self.__class__.__name__

        if self.skip_download_extract():
            log.info("%s: Cache valid, skip downloading & extracting.", cname)
            return

        if isinstance(self.srcs, Iterable):
            for src in self.srcs:
                self.download(src)
                self.extract(src)
        else:
            self.download(self.srcs)
            self.extract(self.srcs)

    @ abstractmethod
    def skip_download_extract(self):
        """If true, will skill the download and extract process"""
        raise NotImplementedError

    def download(self, src: _Source):
        """Download from a source"""
        self.download_path.mkdir(parents=True, exist_ok=True)
        src_path = self.download_path.joinpath(src.file).as_posix()
        if src.url is None:
            log.warning("No url provided")
            if not self.download_path.joinpath(src.file).is_file():
                log.error(
                    "'%s' not found, please download '%s' manually and place "
                    "it under '%s',  and make sure its md5 is %s.",
                    src_path, self.download_path.as_posix(), src.file, src.md5)
                raise Exception("_Source file not found")
        else:
            download_file(src.url, src.md5, src_path)

    def extract(self, src: _Source):
        """Extract the downloaded source"""
        src_path = self.download_path.joinpath(src.file).as_posix()
        if src.compress_type == 'zip':
            with zipfile.ZipFile(src_path, 'r') as zip_ref:
                log.info("Checking '%s' ...", src.file)
                if all(os.path.exists(self.download_path.joinpath(name))
                       for name in zip_ref.namelist()):
                    log.info("'%s' already extracted.", src.file)
                    return
                log.info("Extracting '%s' ...", src.file)
                zip_ref.extractall(self.download_path.as_posix())
        elif src.compress_type == 'tar':
            with tarfile.open(src_path, 'r') as file:
                log.info("Checking '%s' ...", src.file)
                if all(os.path.exists(self.download_path.joinpath(name))
                        for name in file.getnames()):
                    log.info("'%s' already extracted.", src.file)
                    return
                log.info("Extracting '%s' ...", src.file)
                self.safe_extract(file, self.download_path.as_posix())
        elif src.compress_type == 'tgz':
            with tarfile.open(src_path, 'r:gz') as file:
                log.info("Checking '%s' ...", src.file)
                if all(os.path.exists(self.download_path.joinpath(name))
                        for name in file.getnames()):
                    log.info("'%s' already extracted.", src.file)
                    return
                log.info("Extracting '%s' ...", src.file)
                self.safe_extract(file, self.download_path.as_posix())
        elif src.compress_type == 'gz':
            out_file = re.sub('.gz$', '', src.file)
            out_file = (out_file if out_file != src.file
                        else src.file + '.unzipped')
            out_path = self.download_path.joinpath(out_file).as_posix()
            log.info("Extracting '%s' ...", src.file)
            with gzip.open(src_path, 'rb') as fin, open(out_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        else:
            raise Exception("Unknown compression type")

    @staticmethod
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        """CVE-2007-4559. Replaces tar.extract_all(path)"""
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(path, members, numeric_owner=numeric_owner)
