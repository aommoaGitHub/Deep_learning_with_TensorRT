# Tensor RT


## Workflow
https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#tf_rnn_workflow
Dump Weights -> Load Weights -> Convert Weights -> Set Weights

### Dump Weights
Python script dumpTFWts.py can be used to dump all the variables and weights from a given TensorFlow checkpoint. The script is located in the `/usr/src/tensorrt/samples/common/` dumpTFWts.py directory. Issue dumpTFWts.py -h for more information on the usage of this script.

### Load Weights
Function loadWeights() loads from the dump of the dumpTFWts.py script. It has been provided as an example in the Building An RNN Network Layer By Layer sample. The function signature is:
 `std::map<std::string, Weights> loadWeights(const std::string file, std::unordered_set<std::string> names);` <br>
This function loads the weights specified by the names set from the specified file and returns them in a `std::map<std::string, Weights>.` 

### Convert Weights
- At this point, we are ready to convert the weights. To do this, the following steps are required:
Understanding and using the TensorFlow checkpoint to get the tensor.
Understanding and using the tensors to extract and reformat relevant weights and set them to the corresponding layers in TensorRT.
- TensorFlow Checkpoint Storage Format
There are two possible TensorFlow checkpoint storage formats:
Platform independent format - separated by layer
Cell_i_kernel <Weights>
Cell_i_bias <Weights>
cuDNN compatible format - separated by input and recurrent
Cell_i_Candidate_Input_kernel <Weights>
Cell_i_Candidate_Hidden_kernel <Weights>
In other words, 1.1 Cell_i_kernel <Weights> in the concatenation of 2.1 Cell_i_Candidate_Input_kernel <Weights> and 2.2 Cell_i_Candidate_Hidden_kernel <Weights>. Therefore, storage format 2 is simply a more fine-grain version of storage format 1.

- TensorFlow Kernel Tensor Storage Format
Before storing the weights in the checkpoint, TensorFlow transposes and then interleaves the rows of transposed matrices. The order of the interleaving is described in the next section. A figure is provided in BasicLSTMCell Example to further illustrate this format.

- Gate Order Based On Layer Operation Type The transposed weight matrices are interleaved in the following order:
1. RNN RuLU/Tanh: input gate (i)
2. LSTM: input gate (i), cell gate (c) , forget gate (f), output gate (o)
3. GRU: reset (r), update (u)
- Kernel Weights Conversion To A TensorRT Format
Converting the weights from TensorFlow format can be summarized in two steps.
Reshape the weights to push the interleaving down to a lower dimension.
Transpose the weights to get rid of the interleaving completely and have the weight matrices stored contiguously in memory.
Transformation Utilities To help perform these transformations correctly, reorderSubBuffers(), transposeSubBuffers(), and reshapeWeights() are functions that have been provided. For more information, see NvUtils.h.

- TensorFlow Bias Weights Storage Format
The bias tensor is simply stored as contiguous vectors concatenated in the order specified in TensorFlow Kernel Tensor Storage Format. If the checkpoint storage is platform independent, then TensorFlow combines the recurrent and input biases into a single tensor by adding them together. Otherwise, the recurrent and input biases and stored in separate tensors.

- Bias Tensor Conversion To TensorRT Format
Since the biases are stored as contiguous vectors, there aren’t any transformations that need to be applied to get the bias into the TensorRT format.

### Set Weights

## Sources

- https://developer.nvidia.com/tensorrt
- https://github.com/the-house-of-black-and-white/hall-of-faces
Face detection
Area under the ROC curve show how good the model of detection.

## TensorFlow -> TensorRT on Jetson-tx2
- https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification.git

There are python code for converting TensorFlow model to TensorRT model. 

## Simple Run
- https://github.com/KleinYuan/py-TensorRT