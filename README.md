# About PyArmNN

PyArmNN is a python extension for [Arm NN SDK](https://developer.arm.com/ip-products/processors/machine-learning/arm-nn).
PyArmNN provides interface similar to Arm NN C++ Api.

PyArmNN is built around public headers from the armnn/include folder of Arm NN. PyArmNN does not implement any computation kernels itself, all operations are
delegated to the Arm NN library. There is a few things which are reimplemented with Python in mind though.

The following diagram shows the conceptual architecture of this library:
![PyArmNN](./images/pyarmnn.png)

This project contains prebuilt wheel and source packages to be installed using pip. Distributions are currently supported only for Python3.x versions.

# PyArmNN installation

PyArmNN is distributed as a source package or a binary package (wheel). Wheel package for ArmNN 19.08 is the only distribution, which is currently tested.

Install dependencies:
* Numpy (1.18.1)

Runtime dependencies:
* ArmNN libraries (19.08 or 19.11)
* Protocol Buffers (3.5.1)


To install use pip, e.g.:
```bash
$ pip3 install /path/to/pyarmnn-19.8.0-cp37-cp37m-linux_aarch64.whl
```
