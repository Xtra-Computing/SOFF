import os
import shutil
import hashlib
import logging
import subprocess

if os.getenv("SKIP_CERT_VERIFY") is not None:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

log = logging.getLogger(__name__)


def file_exists(path: str):
    return os.path.isfile(path)


def file_already_cached(path: str, md5: str) -> bool:
    # check if already installed and md5sum is correct
    if os.path.isfile(path):
        if md5 is None:
            return True
        else:
            with open(path, 'rb') as f:
                file_hash = hashlib.md5()
                # read 4M chunks
                while chunk := f.read(4 * 1024 * 1024):
                    file_hash.update(chunk)
            if file_hash.hexdigest() == md5:
                return True

    return False


def download_file(url: str, md5: str, path: str) -> bool:
    if file_already_cached(path, md5):
        log.info("file already downloaded ({})".format(path))
        return True
    try:
        if shutil.which('wget') is None:
            raise Exception("'wget' is required for downloading datasets")

        # download using external wget, since there's some issue
        # with the python 'wget' package.
        log.info("downloading {}".format(url))
        res = subprocess.run(['wget', url, '-O', path])
        if res.returncode == 0:
            return True
        else:
            raise Exception(
                "Download failed (return code: {})" .format(res.returncode))
    except Exception as e:
        log.error("Could not download {}: Exception: {}".format(url, e))
        return False
