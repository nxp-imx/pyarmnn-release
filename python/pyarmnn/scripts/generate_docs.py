# Copyright © 2019 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

import os
import tarfile

import pyarmnn as ann
import shutil

from typing import List, Union

from pdoc.cli import main

package_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

def __copy_file_to_dir(file_paths: Union[List[str], str], target_dir_path: str):
    """Copies multiple files to a directory.

    Args:
        file_paths (Union[List(str)]): List of files to copy
        target_dir_path (str): Target directory.

    Returns:
        None
    """
    
    file_paths = [] + file_paths

    if not (os.path.exists(target_dir_path) and os.path.isdir(target_dir_path)):
        os.makedirs(target_dir_path)

    for file_path in file_paths:
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            raise RuntimeError('Not a file: {}'.format(file_path))

        file_name = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(str(target_dir_path), file_name))


def archive_docs(path: str, version: str):
    """Creates an archive.

    Args:
        path (str): Path which will be archived.
        version (str): Version of Arm NN.

    Returns:
        None
    """
    
    output_filename = f'pyarmnn_docs-{version}.tar'

    with tarfile.open(os.path.join(package_dir, output_filename), "w") as tar:
        tar.add(path)


if __name__ == "__main__":
    readme_filename = os.path.join(package_dir, '..', '..', 'README.md')
    with open(readme_filename, 'r') as readme_file:
        top_level_pyarmnn_doc = ''.join(readme_file.readlines())
        ann.__doc__ = top_level_pyarmnn_doc

    main()
    target_path = os.path.join(package_dir, 'docs')
    archive_docs(target_path, ann.__version__)
