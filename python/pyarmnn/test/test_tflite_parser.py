# Copyright © 2019 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os

import pytest
import pyarmnn as ann
import numpy as np


@pytest.fixture()
def parser(shared_data_folder):
    """
    Parse and setup the test network (ssd_mobilenetv1) to be used for the tests below
    """
    parser = ann.ITfLiteParser()
    parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'ssd_mobilenetv1.tflite'))

    yield parser


def test_tflite_parser_swig_destroy():
    assert ann.ITfLiteParser.__swig_destroy__, "There is a swig python destructor defined"
    assert ann.ITfLiteParser.__swig_destroy__.__name__ == "delete_ITfLiteParser"


def test_check_tflite_parser_swig_ownership(parser):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert parser.thisown

def test_tflite_get_sub_graph_count(parser):
    graphs_count = parser.GetSubgraphCount()
    assert graphs_count == 1


def test_tflite_get_network_input_binding_info(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    input_names = parser.GetSubgraphInputTensorNames(graph_id)

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])

    tensor = input_binding_info[1]
    assert tensor.GetDataType() == 2
    assert tensor.GetNumDimensions() == 4
    assert tensor.GetNumElements() == 270000
    assert tensor.GetQuantizationOffset() == 128
    assert tensor.GetQuantizationScale() == 0.0078125


def test_tflite_get_network_output_binding_info(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    output_binding_info1 = parser.GetNetworkOutputBindingInfo(graph_id, output_names[0])
    output_binding_info2 = parser.GetNetworkOutputBindingInfo(graph_id, output_names[1])
    output_binding_info3 = parser.GetNetworkOutputBindingInfo(graph_id, output_names[2])
    output_binding_info4 = parser.GetNetworkOutputBindingInfo(graph_id, output_names[3])

    # Check the tensor info retrieved from GetNetworkOutputBindingInfo
    tensor1 = output_binding_info1[1]
    tensor2 = output_binding_info2[1]
    tensor3 = output_binding_info3[1]
    tensor4 = output_binding_info4[1]

    assert tensor1.GetDataType() == 1
    assert tensor1.GetNumDimensions() == 3
    assert tensor1.GetNumElements() == 40
    assert tensor1.GetQuantizationOffset() == 0
    assert tensor1.GetQuantizationScale() == 0.0

    assert tensor2.GetDataType() == 1
    assert tensor2.GetNumDimensions() == 2
    assert tensor2.GetNumElements() == 10
    assert tensor2.GetQuantizationOffset() == 0
    assert tensor2.GetQuantizationScale() == 0.0

    assert tensor3.GetDataType() == 1
    assert tensor3.GetNumDimensions() == 2
    assert tensor3.GetNumElements() == 10
    assert tensor3.GetQuantizationOffset() == 0
    assert tensor3.GetQuantizationScale() == 0.0

    assert tensor4.GetDataType() == 1
    assert tensor4.GetNumDimensions() == 1
    assert tensor4.GetNumElements() == 1
    assert tensor4.GetQuantizationOffset() == 0
    assert tensor4.GetQuantizationScale() == 0.0


def test_tflite_get_subgraph_input_tensor_names(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    input_names = parser.GetSubgraphInputTensorNames(graph_id)

    assert input_names == ('normalized_input_image_tensor',)


def test_tflite_get_subgraph_output_tensor_names(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    assert output_names[0] == 'TFLite_Detection_PostProcess'
    assert output_names[1] == 'TFLite_Detection_PostProcess:1'
    assert output_names[2] == 'TFLite_Detection_PostProcess:2'
    assert output_names[3] == 'TFLite_Detection_PostProcess:3'


def test_tflite_filenotfound_exception(shared_data_folder):
    parser = ann.ITfLiteParser()

    with pytest.raises(RuntimeError) as err:
        parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'some_unknown_network.tflite'))

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Cannot find the file' in str(err.value)


def test_tflite_parser_end_to_end(shared_data_folder):
    parser = ann.ITfLiteParser()

    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder,"inception_v3_quant.tflite"))

    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert 0 == len(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)
    assert "" == messages

    # Load test image data stored in input.npy
    input_tensor_data = np.load(os.path.join(shared_data_folder, 'tflite_parser/inceptionv3_golden_input.npy'))
    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])

    output_tensors = []
    for index, output_name in enumerate(output_names):
        out_bind_info = parser.GetNetworkOutputBindingInfo(graph_id, output_name)
        out_tensor_info = out_bind_info[1]
        out_tensor_id = out_bind_info[0]
        output_tensors.append((out_tensor_id,
                               ann.Tensor(out_tensor_info)))

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_vectors = []
    for index, out_tensor in enumerate(output_tensors):
        output_vectors.append(out_tensor[1].get_memory_area())

    # Load golden output file to compare the output results with
    expected_outputs = np.load(os.path.join(shared_data_folder, 'tflite_parser/inceptionv3_golden_output.npy'))

    # Check that output matches golden output
    np.testing.assert_allclose(output_vectors, expected_outputs, 0.08)
