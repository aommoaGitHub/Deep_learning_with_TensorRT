# Tensor RT
## Sources
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html
- https://developer.nvidia.com/tensorrt

## Progress
**>>>>>the success file is /notebook/1-Keras2TRT.ipynb and 2-InferenceTRT**<br>

**28-6-19** All code for inferencing Keras Model with TensorRT is done but there are 2 bugs
	1. Cannot optimize the frozen model to TensorRT graph
	2. Cannot inferencing

**3-7-19** Seperate into 2 files 
	1.Converting Keras to frozen graph and then optimize to .pb files
	2.Inferencing<br>
	Bugs of Inferencing is fixed, `It is a problem related to topK parameter in DetectionOutput layer. 5000 is too large that TensorRT crushed during runtime. After I reduce to below 2500 it runs fine.` Ref: https://devtalk.nvidia.com/default/topic/1037616/tensorrt/problems-with-nvidia-ssddetectionoutputplugin/

## Keras Workflow
- https://github.com/jeng1220/KerasToTensorRT -- I cannot run it. Could you solve it?
- https://www.youtube.com/watch?v=AIGOSz2tFP8&list=PLkRkKTC6HZMwdtzv3PYJanRtR6ilSCZ4f

## TensorFlow Workflow
- https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/

### TensorFlow -> TensorRT on Jetson-tx2
- https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification.git

There are python code for converting TensorFlow model to TensorRT model. 

### Accelerating Interference In TF-TRT
- https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html

1. TF-TRT Worlflow with A SavedModel
2. TF-TRT Workflow with A Frozen Graph
3. TF-TRT Workflow with MetaGraph and Checkpoint Files

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
