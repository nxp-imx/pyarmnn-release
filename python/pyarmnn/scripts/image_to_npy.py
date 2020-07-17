# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

import numpy as np
from PIL import Image
import argparse
import os


def imload(filename: str, im_width: int, im_height: int, datatype: str):
    """Converts an image to a numpy array and resizes.

    Args:
        filename (str): Image filename.
        im_width (int): Image width.
        im_height (int): Image height.
        datatype (str): Datatype to convert to (float/uint8). Float scales to <0;1> range.

    Returns:
        np.array: Image as a numpy array.
    """

    img = Image.open(filename)
    img = img.resize((im_width, im_height))
    img_rgb = img.convert('RGB')
    numpy_img_rgb = np.array(img_rgb)

    if datatype == "float":
        numpy_img_rgb = numpy_img_rgb.astype('f') / 255.0
    elif datatype == "uint8":
        numpy_img_rgb = numpy_img_rgb.astype(np.uint8)
    else:
        raise Exception("Unsupported datatype.")
    return numpy_img_rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a tensorflow frozen model to tflite.')
    parser.add_argument('--image', action='store', dest='image_file',
                        help='Input image.')
    parser.add_argument('--output', action='store', dest='output_npy',
                        help='Output NPY file.')
    parser.add_argument('--width', action='store', dest='image_width',
                        help='Image width.')
    parser.add_argument('--height', action='store', dest='image_height',
                        help='Image height.')
    parser.add_argument('--datatype', action='store', dest='datatype', default="float",
                        help='Type of data (float, uint8).')

    args = parser.parse_args()

    np_arr = imload(args.image_file, int(args.image_width), int(args.image_height), args.datatype)
    output_filename = args.output_npy
    filename_base = os.path.basename(args.image_file)
    filename, ext = os.path.splitext(filename_base)
    if output_filename is None:
        output_filename = os.path.join(os.getcwd(), filename)
    else:
        dir_path = os.path.dirname(output_filename)
        # path does not exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # dir and not file is specified
        if os.path.isdir(output_filename):
            output_filename = os.path.join(output_filename, filename)

    np.save(output_filename, np_arr)
