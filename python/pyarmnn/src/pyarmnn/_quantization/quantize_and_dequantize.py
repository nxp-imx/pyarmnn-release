# Copyright © 2019 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT

from .._generated.pyarmnn import Quantize_uint8_t, Quantize_int16_t, Quantize_int32_t, \
    Dequantize_uint8_t, Dequantize_int16_t, Dequantize_int32_t

__dtype_to_quantize_function = {
        'uint8': Quantize_uint8_t,
        'int16': Quantize_int16_t,
        'int32': Quantize_int32_t
    }

__dtype_to_dequantize_function = {
        'uint8': ((0, 255), Dequantize_uint8_t),
        'int16': ((-32768, 32767), Dequantize_int16_t),
        'int32': ((-2147483648, 2147483647), Dequantize_int32_t)
    }


def quantize(value: float, scale: float, offset: int, target_dtype: str) -> int:
    """Quantize given value to the given target datatype using Arm NN.

    This function can be used to convert a 32-bit floating point value into 16/32-bit
    integer or 8-bit unsigned integer values.

    Args:
        value (float): The value to be quantized.
        scale (float): A numeric constant that the value is multiplied by.
        offset (int): A 'zero-point' used to 'shift' the integer range.
        target_dtype (str): The target data type. Supported values: 'unit8', 'int16', 'int32'.

    Returns:
        int: A quantized 8-bit unsigned integer value or 16/32-bit integer value.
    """

    if target_dtype not in __dtype_to_quantize_function:
        raise ValueError("""Unexpected target datatype {} given.
                         Armnn currently supports quantization to {} values.""".format(target_dtype, list(__dtype_to_quantize_function.keys())))

    return __dtype_to_quantize_function[target_dtype](float(value), scale, offset)


def dequantize(value: int, scale: float, offset: float, from_dtype: str) -> float:
    """Dequantize given value from the given datatype using Armnn.

    This function can be used to convert an 8-bit unsigned integer value or 16/32-bit
    integer value into a 32-bit floating point value. Typically used when decoding an
    output value from an output tensor on a quantized model.

    Args:
        value (int): The value to be dequantized. Value could be numpy numeric data type.
        scale (float): A numeric constant that the value is multiplied by.
        offset (float): A 'zero-point' used to 'shift' the integer range.
        from_dtype (str): The data type 'value' represents. Supported values: 'unit8', 'int16', 'int32'.

    Returns:
        float: A dequantized 32-bit floating-point value.
    """

    # specifies which function to use with given datatype and the value range for that data type.
    if from_dtype not in __dtype_to_dequantize_function:
        raise ValueError("""Unexpected value datatype {} given. 
                         Armnn currently supports dequantization from {} values.""".format(from_dtype, list(__dtype_to_dequantize_function.keys())))

    input_range = __dtype_to_dequantize_function[from_dtype][0]

    if not input_range[0] <= value <= input_range[1]:
        raise ValueError('Value is not within range of the given datatype {}'.format(from_dtype))

    return __dtype_to_dequantize_function[from_dtype][1](int(value), scale, offset)
