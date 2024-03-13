"""Dataset-processing related utilities"""
import os
import pickle
import shutil
import hashlib
import pathlib
import logging
import subprocess
import multiprocessing
from typing import Union

# if os.getenv("SKIP_CERT_VERIFY") is not None:
#     import ssl
#     ssl._create_default_https_context = ssl._create_unverified_context

log = logging.getLogger(__name__)


def file_exists(path: Union[str, pathlib.Path]):
    """Test if path exists"""
    return os.path.isfile(path)


# Cache related
def save_obj(obj, cache_path: Union[str, pathlib.Path]):
    """Cache `obj` to `cache_path`"""
    with open(cache_path, 'wb') as file:
        pickle.dump(obj, file)


def load_obj(cache_path: Union[str, pathlib.Path]):
    """Load cache from `cache_path`"""
    with open(cache_path, 'rb') as file:
        return pickle.load(file)


def save_dict(dic: dict, cache_path: Union[str, pathlib.Path]):
    """Cache dictionary `dic` to `cache_path`"""
    assert isinstance(dic, dict)
    save_obj(dic, cache_path)


def metadata_updated(meta, meta_cache_path, key_list=None) -> bool:
    """Check if current metadata is different than the cached one
    If metadata updated, return True, otherwise return False"""
    if not file_exists(meta_cache_path):
        return True
    with open(meta_cache_path, 'rb') as file:
        old_meta = pickle.load(file)
        if key_list is None:
            return meta != old_meta
        assert isinstance(key_list, list)
        return any(key not in old_meta or meta[key] != old_meta[key]
                   for key in key_list if key in meta)


def file_already_cached(path: str, md5: str) -> bool:
    """Check if file already installed and md5sum is correct"""
    if os.path.isfile(path):
        if md5 is None:
            return False

        with open(path, 'rb') as file:
            file_hash = hashlib.md5()
            # read 4M chunks
            while chunk := file.read(4 * 1024 * 1024):
                file_hash.update(chunk)
        if file_hash.hexdigest() == md5:
            return True

    return False


def download_file(url: str, md5: str, path: str) -> bool:
    """Download file from `url` to `path`. Skip if already downloaded"""
    if file_already_cached(path, md5):
        log.info("'%s' already downloaded", path)
        return True

    if shutil.which('wget') is None:
        log.error("'wget' is required for downloading datasets")
        return False

    # download using external wget, there's some issue with the python's 'wget'
    log.info("downloading '%s'", url)
    res = subprocess.run(['wget', url, '-O', path], check=False)
    if res.returncode == 0:
        return True
    log.error("Could not download '%s': %s (%s)", url, res, res.returncode)
    return False


def multiprocess_num_jobs() -> int:
    """Get numer of paralle processes when running multiprocessing jobs"""
    return max(int(os.getenv("NUM_JOBS", multiprocessing.cpu_count() - 1)), 1)
