# Tensor RT

## Keras Workflow
![alt text](https://gitlab.iz.hs-offenburg.de/imla/demos/tensor_rt/blob/vittunyuta/notebook/pictures/Keras_to_TensorRT.png "Keras Workflow")
- https://github.com/jeng1220/KerasToTensorRT -- I cannot run it. Could you solve it?
- https://www.youtube.com/watch?v=AIGOSz2tFP8&list=PLkRkKTC6HZMwdtzv3PYJanRtR6ilSCZ4f

## TensorFlow Workflow
![alt text](https://gitlab.iz.hs-offenburg.de/imla/demos/tensor_rt/blob/vittunyuta/notebook/pictures/tf-trt_workflow.png "Tensorflow Workflow")
- https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/

### 1. TensorFlow -> TensorRT on Jetson-tx2
- https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification.git

There are python code for converting TensorFlow model to TensorRT model. 

### Accelerating Interference In TF-TRT
- https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html

1. TF-TRT Worlflow with A SavedModel
2. TF-TRT Workflow with A Frozen Graph
3. TF-TRT Workflow with MetaGraph and Checkpoint Files

## Sources
- https://developer.nvidia.com/tensorrt

### Hall of face - Face detection model
- https://github.com/the-house-of-black-and-white/hall-of-faces
Area under the ROC curve show how good the model of detection.
(mAP measurements) https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

## Simple Run
- https://github.com/KleinYuan/py-TensorRT
- https://github.com/JerryJiaGit/facenet_trt

## More
- onnx

## Reference
- Workflow Image: https://github.com/ardianumam/Tensorflow-TensorRT.git
- Convert Keras Model to Tensorflow model: https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a
