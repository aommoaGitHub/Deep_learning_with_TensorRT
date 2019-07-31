# TensorRT
[TensorRT](https://developer.nvidia.com/tensorrt) is a deep learning inference platform. It can be integrated with TensorFlow to accelerate inference such as speed up the inference time. There are [3 different ways of integration workflow.](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
1. TF-TRT Workflow with A SavedModel
2. TF-TRT Workflow with A Frozen Graph
3. TF-TRT Workflow with MetaGraph and Checkpoint Files

#### More helpful link about TensorRT
- Workflow: https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/
- Guide: https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html

## Outline
This project is divided into 2 path.
1. Optimize Keras model with TensorRT [here](https://gitlab.com/imla/demos/tensor_rt/tree/vittunyuta/keras)
2. Optimize Object detection model with TensorRT [here](https://gitlab.com/imla/demos/tensor_rt/tree/vittunyuta/object_detection)

## Optimize Keras model with TensorRT
The selected integration workflow is a workflow with a frozen graph. It's concluded into 3 steps.
![Image of Workflow](https://i.imgur.com/2xVQrMl.png)

#### 1. Convert model to frozen model
In this case, the Keras model to be converted is trained Magma model from the previous work. There is a function from TensorFlow named `convert_variables_to_constants` which use to freeze the model. [Sample Code.](https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a) This function can freeze both Keras and TensorFlow model and return a frozen model. **That means both of the Keras and TensorFlow models can be frozen in the same way.** You can use `gflie` library to save a frozen model as a .pb file that allows you to load the model in several times.

#### 2. Optimize frozen model with TensorRT
 This step use `create_inference_graph` function to optimize frozen model to TensorRT model. The function return TensorRT model (graph). Finally, save the a TensorRT model as .pb file.
![Image of Code](https://i.imgur.com/szk1ViC.png)
The [arguments](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api) of optimizing.
- **input_graph_def:** input a frozen graph which is returned by  [**tf.graph_util.convert_variables_to_constants**](https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants) function. ![Image of frozen](https://i.imgur.com/8UkjFrC.png)
Above image is an example code. This function require 3 arguments.
	1. **session:** Active TensorFlow session
	2. **input_graph_def:** TensorFlow GraphDef which is loaded from .pb file
	3. **output_node_names:** List of all output nodes name in the graph.
- **outputs:** List of all output nodes name in the graph.
- **max_batch_size:** the max size for the input batch. That means "How many images you can inference at the same time". The default value is 1.
- **max_work_space:** The maximum GPU temporary memory which the TensorRT engine can use for execution.
- **precision_mode:** It is a data type that the optimized model can have graph and parameters stored in. The available modes are "FP32"(float32),"FP16","INT8". The default value is "FP32".


**Step 1 and 2 are in [1-main-Converting.ipynb](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/keras/1-main-Converting.ipynb)**

#### 3. Inference using TensorRT model
The inference is the stage in which a trained model is used to infer/predict the testing samples. It similar forward pass as training to predict the values. [2-InferenceTRT.ipynb](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/notebook/2-InferenceTRT.ipynb) is a inferencing code file. The steps are
1. load frozen model (.pb file) which has already optimized with TensorRT
2. import the loaded model using import_graph_def
3. Get input and output tensors
4. Write logs for TensorBoard (optional)
5. Inference using function run of TensorFlow Session.

In this case, I inference 50 times to find the average inference time.
You can repeat these steps with the original frozen model (the frozen model without optimizing with TensorRT) for comparison.

### Result
The objective of this path is a comparison between the original model and optimized model by TensorRT. There are some basic comparison and its result.
- Time: Inferencing by the optimized model take less time than the original model. ![Diff Time](https://i.imgur.com/f4dAc0M.png) But if a number test predictions images is low such as 30 images, unoptimized model sometimes might be faster or equal.
- Prediction Result: Both are the same.
![Prediction result](https://i.imgur.com/QHq2rCt.png)
- Others metrics such as Confusion matrix, Recall score, Precision score,  f1-score, ROC curve and AUC (area under curve): Both are the same.

### Important Reference
- Workflow: https://www.youtube.com/watch?v=AIGOSz2tFP8&list=PLkRkKTC6HZMwdtzv3PYJanRtR6ilSCZ4f
- Converting Code: https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a
- Inference Code: https://github.com/ardianumam/Tensorflow-TensorRT.git
- Metrics: https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019

## Optimize Object detection model with TensorRT
The workflow is almost the same as optimizing Keras model. The main difference is **model** for object detection are more diversity and complexity. And the **dataset** is from car camera.

### Directory Structure
+ `object_detection/`
	+ `data/` -> contains label map of models
	+ `logs/` -> logs created while load original model, which used by TensorBoard
	+ `trt_logs/` -> logs created while load optimized model, which used by TensorBoard
	+ `utils/` -> python helper code such as visualization
	+ `test_images2/` -> contain 2 images for detection testing
	+ `models/`
		+ `ssd_mobilenet_v1_coco_2017_11_17/`
			+ frozen_inference_graph.pb
		+ `faster_rcnn_resnet101_kitti_2018_01_28/`
			+ frozen_inference_graph.pb
		+ `aadc2018_frcnn_res101_200k_kitti/` -> given model, dataset, and labels
			+ `test_images_20181027` -> contains test images which is images from the car camera.
			+ `detected_images/` -> contains images after detection
			+ aadc2018_frcnn_res101_200k_kitti.pb
			+ aadc_labels_2018.pbtxt
			+ aadc_labels_2018_without_middlelane.pbtxt
			+ aadc_labels_2018_slim.pbtxt
	+ `InferenceWithTensorRT.ipynb` -> main notebook file

### Models
There are 3 models are considered.
1. `ssd_mobilenet_v1_coco_2017_11_17` is a [Single-Shot multi-box Detection (SSD)](https://medium.com/@smallfishbigsea/understand-ssd-and-implement-your-own-caa3232cd6ad) network intended to perform object detection. This model is a default model in [tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb). So, It's a starting model. You can download from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz).
2. `faster_rcnn_resnet101_kitti_2018_01_28/` is a network for object detection. It has use cases in self-driving cars, manufacturing, security.
3. `aadc2018_frcnn_res101_200k_kitti/` is similar to faster_rcnn_kitti model but more suitable with the dataset.

### Problems and Causes
1. `ssd_mobilenet_v1_coco_2017_11_17` can inference both `test_images2` and `test_images_20181027` because it use less memory for inferencing. But this network is not suitable to detect objects in images from car camera. So, the accuracy is too low.
2. `faster_rcnn_resnet101_kitti_2018_01_28/` cannot inference because Jetson tx2 board has only 8GB memory and inferencing take a lot of memory. So, the inferencing process was automatically killed.
3. `aadc2018_frcnn_res101_200k_kitti/` is a newer version of TensorFlow so it cannot be loaded. (TensorFlow version in Jetson tx2 board is 1.9.0 which is an old version)
4. Because there is only `ssd_mobilenet_v1_coco_2017_11_17` which can inference, it is only the model that optimized with TensorRT. But, the optimized model cannot inference. The inferencing process was automatically killed because the process took too much memory. However, please be aware about argument setting especially **max_batch_size** and **max_work_space**. A large **max_batch_size** will make large consuming memory. If **max_work_space** is exceed, a process will be automatically killed.

### Interest things
#### Precision mode in TensorRT
[Precision mode](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#precision-mode) is one of arguments have to be set in optimizing TensorRT. There are 3 available values, "FP32", "FP16", and "INT8".
Here is the result after trying optimized `ssd_mobilenet_v1_coco_2017_11_17` model with 3 different precision mode.
1. FP32: take optimizing time around 9 minutes. Can load optimized model normally.
2. FP16: take optimizing time around 12 minutes. Can load optimized model normally.
3. INT8: take optimizing time around 16 seconds!! but cannot load the optimized model.

In conclusion, precision mode INT8 take the least time and FP16 take the most time. But the optimized model of precision mode INT8 cannot be loaded. It might be because [the model need to be quantized before optimizing](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#int8-quantization).

Note that, time taking of optimizing depend on free memory space. "Much free space, less time taking".

### Reference
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs this link for configuration of each model



## Daily Progress
**Date: Friday 28th June 2019**
All code for inferencing Keras Model with TensorRT is done but there are 2 bugs
1. Cannot optimize the frozen model to TensorRT graph
2. Cannot inferencing

**Date: Wednesday 3rd July 2019**
Seperated work file into 2 files
Converting Keras to frozen graph and then optimize to .pb files
Inferencing
Found a curse of the inferencing error, It is a problem related to topK parameter in DetectionOutput layer. 5000 is too large that TensorRT crushed during runtime. After I reduce to below 2500 it runs fine. Reference: https://devtalk.nvidia.com/default/topic/1037616/tensorrt/problems-with-nvidia-ssddetectionoutputplugin/

**Date: Thursday 4th July 2019**
Bugs of Creating TensorRt inferencer is fixed. the solution is prevention create duplicate layers in a frozen graph.

**Date: Friday 5th July 2019**
Finished Calculating and comparison between using the original model and optimized model with TensorRT. Available measures are time and accuracy.

**Date: Monday 8th July 2019**
Fix the problems about the limit of images in TensorRT inferencing by increase the batch size while creation TensorRT frozen graph.

**Date: Monday 15th July 2019**
Finish showing graph from pb file on Jupyter notebook. And add sklearn accuracy score and f1-score

**Date: Tuesday 16th July 2019**
Add other metrics such as Confusion matrix, Recall score, Precision score,  f1-score, ROC curve and AUC (area under curve).

**Date: Thursday 18th July 2019**
Add TensorRT Optimization of Object detection

**Date: Friday 19th July 2019**
Add faster_rcnn_resnet101_kitti_2018_01_28 model and its test images. But there is an error while prediction.

**Date: Tuesday 23rd July 2019**
Clean a version control such as reduce .git file by removing all large file from all commits (using command git filter-branch --tree-filter 'rm -f <path/to/file>' -- --all) and clean up unnecessary files (using command git gc --aggressive --prune=now), make sure a repository is correct and up to date.

**Date: Wednesday 24th July 2019**
Add aadc2018_frcnn_res101_200k_kitti model and its test images. But the model cannot import. It may be because the model is a newer version of TensorFlow than TensorFlow version on the Jetson board.

**Date: Thursday 25th July 2019**
Add code for checking annotations(found items) of each image, which is used for calculation accuracy. A cause of the problem while prediction is "Out of memory". A inference process uses too much memory so it is killed.

**Date: Friday 26th July 2019**
Trying different precision mode as FP32, FP16, and INT8 then compared and concluded the difference.

**Date: Wednesday 31st July 2019**
Try to solve the problem of cannot inferencing optimized object detection model by increase max_work_space and decrease max_batch_size. But it's still  not working.

## Moreover Source (not done yet)
### Face detection - Hall of face
- https://github.com/the-house-of-black-and-white/hall-of-faces
Area under the ROC curve show how good the model of detection.
(mAP measurements) https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

### Video - Object Detection
- YOLO Realtim: https://github.com/ardianumam/Tensorflow-TensorRT.git
