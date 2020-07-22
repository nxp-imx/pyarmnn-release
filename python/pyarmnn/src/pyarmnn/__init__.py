# Copyright © 2019 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import inspect
import sys
import logging

from ._generated.pyarmnn_version import GetVersion, GetMajorVersion, GetMinorVersion

# Parsers
try:
    from ._generated.pyarmnn_caffeparser import ICaffeParser
except ImportError as err:
    logger = logging.getLogger(__name__)
    message = "Your ArmNN library instance does not support Caffe models parser functionality. "
    logger.warning(message + "Skipped ICaffeParser import.")
    logger.debug(str(err))


    def ICaffeParser():
        raise RuntimeError(message)

try:
    from ._generated.pyarmnn_onnxparser import IOnnxParser
except ImportError as err:
    logger = logging.getLogger(__name__)
    message = "Your ArmNN library instance does not support Onnx models parser functionality. "
    logger.warning(message + "Skipped IOnnxParser import.")
    logger.debug(str(err))


    def IOnnxParser():
        raise RuntimeError(message)

try:
    from ._generated.pyarmnn_tfparser import ITfParser
except ImportError as err:
    logger = logging.getLogger(__name__)
    message = "Your ArmNN library instance does not support TF models parser functionality. "
    logger.warning(message + "Skipped ITfParser import.")
    logger.debug(str(err))


    def ITfParser():
        raise RuntimeError(message)

try:
    from ._generated.pyarmnn_tfliteparser import ITfLiteParser
except ImportError as err:
    logger = logging.getLogger(__name__)
    message = "Your ArmNN library instance does not support TF lite models parser functionality. "
    logger.warning(message + "Skipped ITfLiteParser import.")
    logger.debug(str(err))


    def ITfLiteParser():
        raise RuntimeError(message)

# Network
from ._generated.pyarmnn import Optimize, OptimizerOptions, IOptimizedNetwork, IInputSlot, \
    IOutputSlot, IConnectableLayer, INetwork

# Backend
from ._generated.pyarmnn import BackendId
from ._generated.pyarmnn import IDeviceSpec

# Tensors
from ._generated.pyarmnn import TensorInfo, TensorShape

# Runtime
from ._generated.pyarmnn import IRuntime, CreationOptions

# Profiler
from ._generated.pyarmnn import IProfiler

# Types
from ._generated.pyarmnn import DataType_Float16, DataType_Float32, DataType_QuantisedAsymm8, DataType_Signed32, \
    DataType_Boolean, DataType_QuantisedSymm16
from ._generated.pyarmnn import DataLayout_NCHW, DataLayout_NHWC
from ._generated.pyarmnn import ActivationFunction_Abs, ActivationFunction_BoundedReLu, ActivationFunction_LeakyReLu, \
    ActivationFunction_Linear, ActivationFunction_ReLu, ActivationFunction_Sigmoid, ActivationFunction_SoftReLu, \
    ActivationFunction_Sqrt, ActivationFunction_Square, ActivationFunction_TanH, ActivationDescriptor
from ._generated.pyarmnn import BatchNormalizationDescriptor, BatchToSpaceNdDescriptor
from ._generated.pyarmnn import Convolution2dDescriptor, DepthwiseConvolution2dDescriptor, \
    DetectionPostProcessDescriptor, FakeQuantizationDescriptor, FullyConnectedDescriptor, \
    LstmDescriptor, L2NormalizationDescriptor, MeanDescriptor
from ._generated.pyarmnn import NormalizationAlgorithmChannel_Across, NormalizationAlgorithmChannel_Within, \
    NormalizationAlgorithmMethod_LocalBrightness, NormalizationAlgorithmMethod_LocalContrast, NormalizationDescriptor
from ._generated.pyarmnn import PadDescriptor
from ._generated.pyarmnn import PermutationVector, PermuteDescriptor
from ._generated.pyarmnn import OutputShapeRounding_Ceiling, OutputShapeRounding_Floor, \
    PaddingMethod_Exclude, PaddingMethod_IgnoreValue, PoolingAlgorithm_Average, PoolingAlgorithm_L2, \
    PoolingAlgorithm_Max, Pooling2dDescriptor
from ._generated.pyarmnn import ResizeMethod_Bilinear, ResizeMethod_NearestNeighbor, ResizeDescriptor, \
    ReshapeDescriptor, SpaceToBatchNdDescriptor, SpaceToDepthDescriptor, \
    StackDescriptor, StridedSliceDescriptor, SoftmaxDescriptor, TransposeConvolution2dDescriptor, \
    SplitterDescriptor
from ._generated.pyarmnn import ConcatDescriptor, CreateDescriptorForConcatenation

from ._generated.pyarmnn import LstmInputParams

# Public API
# Quantization
from ._quantization.quantize_and_dequantize import quantize, dequantize

# Tensor
from ._tensor.tensor import Tensor
from ._tensor.const_tensor import ConstTensor
from ._tensor.workload_tensors import make_input_tensors, make_output_tensors, workload_tensors_to_ndarray

# Utilities
from ._utilities.profiling_helper import ProfilerData, get_profiling_data

from ._version import __version__, __arm_ml_version__

ARMNN_VERSION = GetVersion()


def __check_version():
    from ._version import check_armnn_version
    check_armnn_version(ARMNN_VERSION)


__check_version()

__all__ = []

__private_api_names = ['__check_version']

for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) or inspect.isfunction(obj):
        if name not in __private_api_names:
            __all__.append(name)
