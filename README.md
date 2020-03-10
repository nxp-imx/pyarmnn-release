# About PyArmNN

PyArmNN is a python extension for [Arm NN SDK](https://developer.arm.com/ip-products/processors/machine-learning/arm-nn).
PyArmNN provides interface similar to Arm NN C++ Api.

PyArmNN is built around public headers from the armnn/include folder of Arm NN. PyArmNN does not implement any computation kernels itself, all operations are
delegated to the Arm NN library.

PyArmNN is currently distributed as a whl package (or also called a binary package), it is based on a development branch from https://review.mlplatform.org/admin/repos/ml/armnn, where you can also find the latest source codes.
