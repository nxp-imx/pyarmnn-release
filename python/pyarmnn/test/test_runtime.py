# Copyright © 2019 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os

import pytest
import numpy as np
from PIL import Image
import pyarmnn as ann
import platform


@pytest.fixture(scope="function")
def random_runtime(shared_data_folder):
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'ssd_mobilenetv1.tflite'))
    preferred_backends = [ann.BackendId('CpuRef')]
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    graphs_count = parser.GetSubgraphCount()

    graph_id = graphs_count - 1
    input_names = parser.GetSubgraphInputTensorNames(graph_id)

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]

    input_tensor_info = input_binding_info[1]

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    input_data = np.random.randint(255, size=input_tensor_info.GetNumElements(), dtype=np.uint8)

    const_tensor_pair = (input_tensor_id, ann.ConstTensor(input_tensor_info, input_data))

    input_tensors = [const_tensor_pair]

    output_tensors = []

    for index, output_name in enumerate(output_names):
        out_bind_info = parser.GetNetworkOutputBindingInfo(graph_id, output_name)

        out_tensor_info = out_bind_info[1]
        out_tensor_id = out_bind_info[0]

        output_tensors.append((out_tensor_id,
                               ann.Tensor(out_tensor_info)))

    yield preferred_backends, network, runtime, input_tensors, output_tensors


@pytest.fixture(scope='function')
def mobilenet_ssd_runtime(shared_data_folder):
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'ssd_mobilenetv1.tflite'))
    graph_id = 0

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, "normalized_input_image_tensor")

    input_tensor_data = np.array(Image.open(os.path.join(shared_data_folder, 'cococat.jpeg')).resize((300, 300)), dtype=np.uint8)

    preferred_backends = [ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    print(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)

    print(messages)

    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    outputs_binding_info = []

    for output_name in output_names:
        outputs_binding_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))

    output_tensors = ann.make_output_tensors(outputs_binding_info)

    yield runtime, net_id, input_tensors, output_tensors


@pytest.mark.skip(reason="IOptimizedNetwork dtor caused segfault, thus was replaced by default. We need to look into why.")
def test_python_disowns_network(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    opt_network, _ = ann.Optimize(network, preferred_backends,
                                                      runtime.GetDeviceSpec(), ann.OptimizerOptions())

    runtime.LoadNetwork(opt_network)

    assert not opt_network.thisown


def test_load_network(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                                      runtime.GetDeviceSpec(), ann.OptimizerOptions())

    net_id, messages = runtime.LoadNetwork(opt_network)
    assert "" == messages
    assert net_id == 0


def test_unload_network_fails_for_invalid_net_id(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]

    ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    with pytest.raises(RuntimeError) as err:
        runtime.UnloadNetwork(9)

    expected_error_message = "Failed to unload network."
    assert expected_error_message in str(err.value)


def test_enqueue_workload(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    input_tensors = random_runtime[3]
    output_tensors = random_runtime[4]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                                      runtime.GetDeviceSpec(), ann.OptimizerOptions())

    net_id, _ = runtime.LoadNetwork(opt_network)
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)


def test_enqueue_workload_fails_with_empty_input_tensors(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    input_tensors = []
    output_tensors = random_runtime[4]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                                      runtime.GetDeviceSpec(), ann.OptimizerOptions())

    net_id, _ = runtime.LoadNetwork(opt_network)
    with pytest.raises(RuntimeError) as err:
        runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    expected_error_message = "Number of inputs provided does not match network."
    assert expected_error_message in str(err.value)


@pytest.mark.skipif(platform.processor() != 'x86_64', reason="Only run on x86, this is because these are exact results "
                                                             "for x86 only. The Juno produces slightly different "
                                                             "results meaning this test would fail.")
@pytest.mark.parametrize('count', [5])
def test_multiple_inference_runs_yield_same_result(count, mobilenet_ssd_runtime):
    """
    Test that results remain consistent among multiple runs of the same inference.
    """
    runtime = mobilenet_ssd_runtime[0]
    net_id = mobilenet_ssd_runtime[1]
    input_tensors = mobilenet_ssd_runtime[2]
    output_tensors = mobilenet_ssd_runtime[3]

    expected_results = [[0.17047899961471558, 0.22598055005073547, 0.8146906495094299, 0.7677907943725586,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0],
                        [16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.80078125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0]]

    for _ in range(count):
        runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

        output_vectors = ann.workload_tensors_to_ndarray(output_tensors)

        for i in range(len(expected_results)):
            assert all(output_vectors[i] == expected_results[i])


@pytest.mark.juno
def test_juno_inference_results(mobilenet_ssd_runtime):
    """
    Test inference results are sensible on a Juno.
    For the Juno we allow +/-3% compared to the results on x86.
    """
    runtime = mobilenet_ssd_runtime[0]
    net_id = mobilenet_ssd_runtime[1]
    input_tensors = mobilenet_ssd_runtime[2]
    output_tensors = mobilenet_ssd_runtime[3]

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_vectors = ann.workload_tensors_to_ndarray(output_tensors)

    expected_outputs = [[pytest.approx(0.17047899961471558, 0.03), pytest.approx(0.22598055005073547, 0.03),
                         pytest.approx(0.8146906495094299, 0.03), pytest.approx(0.7677907943725586, 0.03),
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0],
                        [16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.80078125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0]]

    for i in range(len(expected_outputs)):
        assert all(output_vectors[i] == expected_outputs[i])


def test_enqueue_workload_with_profiler(random_runtime):
    """
    Tests ArmNN's profiling extension
    """
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    input_tensors = random_runtime[3]
    output_tensors = random_runtime[4]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())
    net_id, _ = runtime.LoadNetwork(opt_network)

    profiler = runtime.GetProfiler(net_id)
    # By default profiling should be turned off:
    assert profiler.IsProfilingEnabled() is False

    # Enable profiling:
    profiler.EnableProfiling(True)
    assert profiler.IsProfilingEnabled() is True

    # Run the inference:
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    # Get profile output as a string:
    str_profile = profiler.as_json()

    # Verify that certain markers are present:
    assert len(str_profile) != 0
    assert str_profile.find('\"ArmNN\": {') > 0

    # Get events analysis output as a string:
    str_events_analysis = profiler.event_log()

    assert "Event Sequence - Name | Duration (ms) | Start (ms) | Stop (ms) | Device" in str_events_analysis

    assert profiler.thisown == 0


def test_check_runtime_swig_ownership(random_runtime):
    # Check to see that SWIG has ownership for runtime. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    runtime = random_runtime[2]
    assert runtime.thisown
