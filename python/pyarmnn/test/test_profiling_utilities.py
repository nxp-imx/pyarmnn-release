# Copyright © 2019 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os

import pytest

import pyarmnn as ann


class MockIProfiler:
    def __init__(self, json_string):
        self._profile_json = json_string

    def as_json(self):
        return self._profile_json


@pytest.fixture()
def mock_profiler(shared_data_folder):
    path_to_file = os.path.join(shared_data_folder, 'profile_out.json')
    with open(path_to_file, 'r') as file:
        profiler_output = file.read()
        return MockIProfiler(profiler_output)

@pytest.mark.skip(reason="No way to generate profile_out.json")
def test_inference_exec(mock_profiler):
    profiling_data_obj = ann.get_profiling_data(mock_profiler)

    assert (len(profiling_data_obj.inference_data) > 0)
    assert (len(profiling_data_obj.per_workload_execution_data) > 0)

    # Check each total execution time
    assert (profiling_data_obj.inference_data["execution_time"] == [16035243.953000, 16096248.590000, 16138614.290000,
                                                                    16140544.388000, 16228118.274000, 16543585.760000])
    assert (profiling_data_obj.inference_data["time_unit"] == "us")


@pytest.mark.parametrize("exec_times, unit, backend, workload", [([1233915.166, 1221125.149,
                                                                   1228359.494, 1235065.662,
                                                                   1244369.694, 1240633.922],
                                                                  'us',
                                                                  'CpuRef',
                                                                  'RefConvolution2dWorkload_Execute_#25'),
                                                                 ([270.64, 256.379,
                                                                   269.664, 259.449,
                                                                   266.65, 277.05],
                                                                  'us',
                                                                  'CpuAcc',
                                                                  'NeonActivationWorkload_Execute_#70'),
                                                                 ([715.474, 729.23,
                                                                   711.325, 729.151,
                                                                   741.231, 729.702],
                                                                  'us',
                                                                  'GpuAcc',
                                                                  'ClConvolution2dWorkload_Execute_#80')
                                                                 ])
@pytest.mark.skip(reason="No way to generate profile_out.json")
def test_profiler_workloads(mock_profiler, exec_times, unit, backend, workload):
    profiling_data_obj = ann.get_profiling_data(mock_profiler)

    work_load_exec = profiling_data_obj.per_workload_execution_data[workload]
    assert work_load_exec["execution_time"] == exec_times
    assert work_load_exec["time_unit"] == unit
    assert work_load_exec["backend"] == backend
