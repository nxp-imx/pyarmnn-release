//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/IRuntime.hpp"
#include <iostream>
#include <ostream>
#include <sstream>
%}

namespace std {
    %template() pair<int, string>;
    %template(IntPair) pair<int, int>;
    %template(ConstTensorPair) pair<int, armnn::ConstTensor>;
    %template(TensorPair) pair<int, armnn::Tensor>;

    %template(InputTensorsVector) vector<pair<int, armnn::ConstTensor>>;
    %template(OutputTensorsVector) vector<pair<int, armnn::Tensor>>;
}

%include <std_shared_ptr.i>

%shared_ptr(IGpuAccTunedParameters);

#pragma SWIG nowarn=SWIGWARN_PARSE_NESTED_CLASS

%{
typedef armnn::IRuntime::CreationOptions CreationOptions;
%}

struct CreationOptions
{
    %feature("docstring",
    "
    Structure for holding creation options. For majority of cases it is fine to leave values at default.

    Contains:
        m_GpuAccTunedParameters (IGpuAccTunedParameters): If set, uses the GpuAcc tuned parameters from the given object
                                                          when executing GPU workloads. It will also be updated with new
                                                          tuned parameters if it is configured to do so.

        m_EnableGpuProfiling (bool): Setting this flag will allow the user to obtain GPU profiling information from
                                     the runtime.

        m_DynamicBackendsPath (string): Setting this value will override the paths set by the DYNAMIC_BACKEND_PATHS
                                        compiler directive. Only a single path is allowed for the override.

    ") CreationOptions;

    CreationOptions();
    std::shared_ptr<armnn::IGpuAccTunedParameters> m_GpuAccTunedParameters;
    bool m_EnableGpuProfiling;
    std::string m_DynamicBackendsPath;
};

namespace armnn
{

struct INetworkProperties
{
    %feature("docstring",
    "
    Structure for holding network properties.

    Contains:
        m_ImportEnabled (bool): Enable import.

        m_ExportEnabled (bool): Enable export.

    ") INetworkProperties;
    INetworkProperties(bool importEnabled = false, bool exportEnabled = false);

    const bool m_ImportEnabled;
    const bool m_ExportEnabled;
};

%feature("docstring",
"
Interface for runtime objects.

Runtime objects are responsible for performing inference on an `IOptimizedNetwork`.

Args:
    options (CreationOptions): CreationOptions data struct.

") IRuntime;
%nodefaultctor IRuntime;
class IRuntime
{
public:

    %ignore
    armnn::IRuntime::UnloadNetwork(NetworkId networkId);

    %ignore
    armnn::IRuntime::EnqueueWorkload(NetworkId networkId,
        const std::vector<std::pair<int, armnn::ConstTensor>>& inputTensors,
        const std::vector<std::pair<int, armnn::Tensor>>& outputTensors);

    %feature("docstring",
    "
    Get information relating to networks input tensor.

    Args:
        networkId (int): Unique ID of the network being run.
        layerId (int): Unique ID of the input layer.

    Returns:
        TensorInfo: Information relating to the input tensor a network.
    ") GetInputTensorInfo;
    armnn::TensorInfo GetInputTensorInfo(int networkId, int layerId);

    %feature("docstring",
    "
    Get information relating to networks output tensor.

    Args:
        networkId (int): Unique ID of the network being run.
        layerId (int): Unique ID of the output layer.

    Returns:
        TensorInfo: Information relating to the output tensor a network.
    ") GetOutputTensorInfo;
    armnn::TensorInfo GetOutputTensorInfo(int networkId, int layerId);

    %feature("docstring",
    "
    Get information relating supported compute backends on current device.

    Returns:
        IDeviceSpec: Device spec information detailing all supported backends on current platform.
    ") GetDeviceSpec;
    const IDeviceSpec& GetDeviceSpec();
};

%extend IRuntime {
    //tell python to disown the IOptimizedNetwork pointer
    //because IRuntime takes ownership
    %typemap(in) armnn::IOptimizedNetwork*  {
      if (!SWIG_IsOK(SWIG_ConvertPtr($input, (void **) &$1, $1_descriptor, SWIG_POINTER_DISOWN))) {
        SWIG_exception_fail(SWIG_TypeError, "in method '$symname', argument 2 of type armnn::IOptimizedNetwork*");
      }
    }

    %feature("docstring",
        "
        Loads a complete network into the IRuntime.
        The runtime takes ownership of the network once passed in.
        Args:
            network (IOptimizedNetwork): An optimized network to load into the IRuntime.
            networkProperties (INetworkProperties): Properties that allows the user to opt-in to import/export behavior. Default: None.
        Returns:
            tuple: (int, str) Network id and non fatal failure or warning messsages.
        Raises:
            RuntimeError: If process fails.
        ") LoadNetwork;

    std::pair<int, std::string> LoadNetwork(armnn::IOptimizedNetwork* network,
                                            const INetworkProperties* networkProperties = nullptr)
    {
        armnn::IOptimizedNetworkPtr netPtr(network, &armnn::IOptimizedNetwork::Destroy);
        armnn::NetworkId networkIdOut;
        std::string errorString;
        armnn::Status status;

        if (networkProperties) {
            status = $self->LoadNetwork(networkIdOut, std::move(netPtr), errorString, *networkProperties);
        } else {
            status = $self->LoadNetwork(networkIdOut, std::move(netPtr), errorString);
        }

        if(status == armnn::Status::Failure)
        {
            throw armnn::Exception(errorString);
        }

        auto net_id_int = static_cast<int>(networkIdOut);
        return std::make_pair(net_id_int, errorString);
    };

    %typemap(in) armnn::IOptimizedNetwork*;
    %feature("docstring",
    "
    Calling this function will perform an inference on your network.

    Args:
        networkId (int): Unique ID of the network to run.
        inputTensors (list): A list of tuples (int, `ConstTensor`), see `make_input_tensors`.
        outputTensors (list): A list of tuples (int, `Tensor`), see `make_output_tensors`.

    ") EnqueueWorkload;
    void EnqueueWorkload(int networkId, const std::vector<std::pair<int, armnn::ConstTensor>>& inputTensors,
                         const std::vector<std::pair<int, armnn::Tensor>>& outputTensors) {
        armnn::Status status = $self->EnqueueWorkload(networkId, inputTensors, outputTensors);

        if(status == armnn::Status::Failure)
        {
            throw armnn::Exception("Failed to enqueue workload for network.");
        }
    };

    %feature("docstring",
    "
    Unload a currently loaded network from the runtime.

    Args:
        networkId (int): Unique ID of the network to unload.

    ") UnloadNetwork;
    void UnloadNetwork(int networkId) {
        armnn::Status status = $self->UnloadNetwork(networkId);
        if(status == armnn::Status::Failure)
        {
            throw armnn::Exception("Failed to unload network.");
        }
    };

    %feature("docstring",
    "
    Returns the IProfiler instance registered against the working thread, and stored on the loaded network.
    Be aware that if the runtime has unloaded the network, or if the runtime is destroyed,
    that the IProfiler instance will also be destroyed, and will cause a segmentation fault.

    Args:
        networkId (int): The ID of the loaded network you want to profile.

    Returns:
        IProfiler: IProfiler instance the given loaded network has stored.

    Raises:
        RuntimeError: If no profiler is found.
    ") GetProfiler;

    armnn::IProfiler* GetProfiler(int networkId) {
        std::shared_ptr<armnn::IProfiler> profiler = $self->GetProfiler(networkId);
	if (nullptr == profiler) {
            throw armnn::Exception("Failed to get profiler");
        }
        return profiler.get();
    };

    ~IRuntime() {
        armnn::IRuntime::Destroy($self);
    }

    IRuntime(const CreationOptions& options) {
        return armnn::IRuntime::CreateRaw(options);
    }

}

}

