# About PyArmNN

PyArmNN is a python extension for [Arm NN SDK](https://developer.arm.com/ip-products/processors/machine-learning/arm-nn).
PyArmNN provides interface similar to Arm NN C++ Api.

PyArmNN is built around public headers from the armnn/include folder of Arm NN. PyArmNN does not implement any computation kernels itself, all operations are
delegated to the Arm NN library.

The following diagram shows the conceptual architecture of this library:
![PyArmNN](./docs/images/pyarmnn.png)

PyArmNN is currently distributed as a whl package (or also called a binary package), it is based on a development branch from https://review.mlplatform.org/admin/repos/ml/armnn, where you can also find the latest source codes.

# PyArmNN installation

Binary package is platform dependent and the name of the package will indicate the platform it was built for. It also depends on the version of Python, e.g.:

* Arm NN 19.08 package for Python3.7/Linux Aarch 64 bit machine: pyarmnn-19.8.0-cp37-cp37m-linux_aarch64.whl

The binary package requires the Arm NN library to be present on the target/build machine.

PyArmNN also depends on Numpy python library. It will be automatically downloaded and installed alongside PyArmNN. If your machine does not have access to Python pip repository you might need to install Numpy in advance by following public instructions: https://scipy.org/install.html

## Installing from wheel

Make sure that Arm NN binaries and Arm NN dependencies are installed and can be found in one of the system default library locations. You can check default locations by executing the following command:
```bash
$ gcc --print-search-dirs
```
Install PyArmNN from binary by pointing to the wheel file:
```bash
$ pip3 install /path/to/pyarmnn-19.8.0-cp37-cp37m-linux_aarch64.whl
```

# PyArmNN API overview

#### Getting started
The easiest way to begin using PyArmNN is by using the Parsers. We will demonstrate how to use them below:

Create a parser object and load your model file.
```python
import pyarmnn as ann
import imageio

# ONNX, Caffe and TF parsers also exist.
parser = ann.ITfLiteParser()  
network = parser.CreateNetworkFromBinaryFile('./model.tflite')
```

Get the input binding information by using the name of the input layer.
```python
input_binding_info = parser.GetNetworkInputBindingInfo(0, 'model/input')

# Create a runtime object that will perform inference.
options = ann.CreationOptions()
runtime = ann.IRuntime(options)
```
Choose preferred backends for execution and optimize the network.
```python
# Backend choices earlier in the list have higher preference.
preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

# Load the optimized network into the runtime.
net_id, _ = runtime.LoadNetwork(opt_network)
```
Make workload tensors using input and output binding information.
```python
# Load an image and create an inputTensor for inference.
img = imageio.imread('./image.png')
input_tensors = ann.make_input_tensors([input_binding_info], [img])

# Get output binding information for an output layer by using the layer name.
output_binding_info = parser.GetNetworkOutputBindingInfo(0, 'model/output')
output_tensors = ann.make_output_tensors([outputs_binding_info])
```

Perform inference and get the results back into a numpy array.
```python
runtime.EnqueueWorkload(0, input_tensors, output_tensors)

results = ann.workload_tensors_to_ndarray(output_tensors)
print(results)
```

#### Running examples

For a more complete Arm NN experience, there is a couple of examples located in the examples folder, which require requests, PIL and maybe some other python modules. You may install those using pip.

To run these examples you may simply execute them using the python interpreter. There are no arguments and the resources are downloaded by the scripts:

```bash
$ python3 /path/to/examples/tflite_mobilenetv1_quantized.py
```

*example_utils.py* is a file containg common functions for the rest of the scripts and ot does not execute anything on its own. The rest of the scripts are the examples.
