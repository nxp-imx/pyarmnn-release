# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

import requests
from urllib.parse import urlparse
import os
from zipfile import ZipFile
import tarfile


def download_file(url: str, destination: str = os.getcwd()):
    """Downloads a file.

    Args:
        url (str): File url.
        destination (str): Destination folder.

    Returns:
        str: Path to the downloaded file.
    """

    if not (os.path.exists(destination) and os.path.isdir(destination)):
        os.makedirs(destination)

    path = os.path.join(destination, os.path.basename(urlparse(url).path))
    print("Downloading '{0}' from '{1}' ...".format(path, url))
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)
    print("Finished.")
    return path


def extract_and_remove(archive_path: str, src: str = None, destination: str = os.getcwd()):
    """Extracts a specific file from an archive (tar.gz or zip) or the whole archive.

    Args:
        archive_path (str): Path to the archive.
        src (str): Specific file to extract.
        destination (str): Destination folder.

    Returns:
        str: Path to the extracted file (specific file or archive), None if nothing
    """

    if not (os.path.exists(destination) and os.path.isdir(destination)):
        os.makedirs(destination)

    filename_base = os.path.basename(archive_path)
    filename, ext = os.path.splitext(filename_base)

    extracted_file = None

    if src is not None:
        print("Extracting '{0}' from '{1}' ...".format(src, archive_path))
    else:
        print("Extracting '{0}' ...".format(archive_path))

    if ext == ".zip":
        with ZipFile(archive_path, 'r') as zip_obj:
            if src is None:
                zip_obj.extractall(path=destination)
                extracted_file = archive_path
            else:
                zip_obj.extract(src, path=destination)
                extracted_file = src
    elif ext == ".tar.gz" or ext == ".tgz":
        with tarfile.open(name=archive_path, mode='r:gz') as tar_obj:
            if src is None:
                tar_obj.extractall(path=destination)
                extracted_file = archive_path
            else:
                for member in tar_obj.getmembers():
                    member_filename = os.path.basename(member.name)
                    if member_filename == src:
                        tar_obj.extract(member, path=destination)
                        extracted_file = member_filename
                        break

    print("Finished.")
    print("Removing '{0}' ...".format(archive_path))
    os.remove(archive_path)
    print("Finished.")
    return os.path.join(destination, extracted_file) if extracted_file is not None else None


if __name__ == "__main__":
    dest_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'test', 'testdata', 'shared')
    archive = download_file\
        ('https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/'
         'inception_v3_quant.tgz',
         dest_path)
    extract_and_remove(archive, 'inception_v3_quant.tflite', dest_path)
    archive = download_file\
        ('https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
         'mobilenet_v1_1.0_224.tgz',
         dest_path)
    filename = extract_and_remove(archive, 'mobilenet_v1_1.0_224_frozen.pb', dest_path)
    os.rename(filename, os.path.join(dest_path, 'mobilenet_v1_1.0_224.pb'))
    download_file('https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx',
                  dest_path)
    download_file('https://github.com/forresti/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel',
                  dest_path)
    archive = download_file\
        ('https://storage.googleapis.com/download.tensorflow.org/models/tflite/'
         'coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip',
         dest_path)
    filename = extract_and_remove(archive, 'detect.tflite', dest_path)
    os.rename(filename, os.path.join(dest_path, 'ssd_mobilenetv1.tflite'))
