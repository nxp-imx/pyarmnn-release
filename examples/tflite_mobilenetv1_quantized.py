# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

import pyarmnn as ann
import numpy as np
import example_utils as eu
from zipfile import ZipFile

archive_filename = eu.download_file('https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip')
with ZipFile(archive_filename, 'r') as zip_obj:
   zip_obj.extractall()
labels_filename = 'labels_mobilenet_quant_v1_224.txt'
model_filename = 'mobilenet_v1_1.0_224_quant.tflite'
image_filename = eu.download_file('https://s3.amazonaws.com/model-server/inputs/kitten.jpg')

# Create a network from a model file
net_id, graph_id, parser, runtime = eu.create_tflite_network(model_filename)

# Load input information from the model and create input tensors
input_names = parser.GetSubgraphInputTensorNames(graph_id)
assert len(input_names) == 1 # there is only 1 input tensor
input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
input_width = input_binding_info[1].GetShape()[1]
input_height = input_binding_info[1].GetShape()[2]

# Load output information from the model and create output tensors
output_names = parser.GetSubgraphOutputTensorNames(graph_id)
assert len(output_names) == 1 # and one output tensor
output_bind_info = parser.GetNetworkOutputBindingInfo(graph_id, output_names[0])
output_tensor_id, output_tensor_info = output_bind_info
output_tensors = [(output_tensor_id, ann.Tensor(output_tensor_info))]

# Load labels file
labels = eu.load_labels(labels_filename)

# Load images and resize to expected size
image_names = [ image_filename ]
images = eu.load_images(image_names , input_width, input_height)

for idx, im in enumerate(images):
    # Create input tensors
    input_tensors = ann.make_input_tensors([input_binding_info], [im])

    # Run inference
    print("Running inference on '{0}' ...".format(image_names[idx]))
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)
    
    # Process output
    out_tensor = output_tensors[0][1].get_memory_area()
    results = np.argsort(out_tensor)[::-1]
    eu.print_top_n(5, results, labels, out_tensor)
