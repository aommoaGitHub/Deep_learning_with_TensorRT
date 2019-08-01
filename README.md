# TensorRT
[TensorRT](https://developer.nvidia.com/tensorrt) is a deep learning inference platform. It can be integrated with TensorFlow to accelerate inference such as speed up the inference time. There are [3 different ways of integration workflow.](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
1. TF-TRT Workflow with A SavedModel
2. TF-TRT Workflow with A Frozen Graph
3. TF-TRT Workflow with MetaGraph and Checkpoint Files

#### More helpful link about TensorRT
- Workflow: https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/
- Guide: https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html

### More Definitions
1. **TensorFlow (TF):** a Python library used in production for deep learning models.
	- **TensorFlow Tensor:** Tensor represent all the data in any type and dimensions. The flow of the Tensors refer to the computational graph.
	- **TensorFlow Graph:** Graph show map of the tensor. Graph consists of edges and nodes. Each node called “operation”. There are input nodes, middle level nodes (nodes between input and output nodes), and output nodes. Each node can have either an input or output data(tensor). Input data can be “variables” or “constant”.
	- **TensorFlow Session:** Session is a place where the graph is executed. Technically, session place on hardware such as CPUS or GPUs and provide function for execution.
![tensorflow](https://i.imgur.com/YcNzmA6.png)
2. **Keras:** high-level neural networks Python library that built on TensorFlow which is more user-friendly and easy to use but less advanced operations as compared to TensorFlow.
3. **TensorBoard:** is a suite of web application which used to inspect and understand TensorFlow runs and graph. To use the TensorBoard, run command
	>tensorboard --logdir=/path/to/logs/file/

	For example, this image show command of TensorBoard. Each orange sentence is each graph that found in a directory. Link of the TensorBoard web application is localhost:6006/ or link in the first line of the command’s result.
	![TensorBoard](https://i.imgur.com/dXEJXpv.png)
	You can find the input or output nodes name by considering the detail of the selected node on the top-right of the TensorBoard web application. The input nodes don’t have any input tensors as they're the input themselves. In the same way, the output nodes don’t have any output tensors as they're the output themselves.
	![TensorBoard](https://i.imgur.com/Lex7Bnq.png)![TensorBoard](https://i.imgur.com/PeVnr67.png)

## Outline
This project is divided into 2 path.
1. Optimize Keras model with TensorRT [here](https://gitlab.com/imla/demos/tensor_rt/tree/vittunyuta/keras)
2. Optimize Object detection model with TensorRT [here](https://gitlab.com/imla/demos/tensor_rt/tree/vittunyuta/object_detection)

## Optimize Keras model with TensorRT
The objective of this path is optimizing the Keras model with TensorRT and comparison time, accuracy, other metrics between the original Keras model and optimized model. The selected integration workflow is a workflow with a frozen graph. It's concluded into 3 steps.
![Image of Workflow](https://i.imgur.com/2xVQrMl.png)

#### 1. Convert model to frozen model
In this case, I converted trained Keras Magma model. There is a function from TensorFlow named [`tf.graph_util.convert_variables_to_constants`](https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants) which use to freeze the model. This function can freeze both Keras and TensorFlow model and return a frozen model. This image is a new function using to freeze model. [Ref](https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a)
![Image of frozen](https://i.imgur.com/tUAXCEz.png)

This function require 3 arguments.
1. **session:** Active TensorFlow session
2. **input_graph_def:** TensorFlow GraphDef which is loaded from .pb file
3. **output_node_names:** List of all output nodes name in the graph.
4. **variable_names_whitelist:** (Optional) The set of variable names to convert. By default, all variables are converted or it value is None.
5. **variable_names_blacklist:** (Optional) The set of variable names to omit converting to constants. The default is None.

**That means both of the Keras and TensorFlow models can be frozen in the same way.** You can use `gflie` library to save a frozen model as a .pb file that allows you to load the model in several times. ![gfile](https://i.imgur.com/8TcLkid.png) So now, you have *a frozen Keras model.*

#### 2. Optimize frozen model with TensorRT
This step use `create_inference_graph` function to optimize frozen model with TensorRT. The function return TensorRT model (graph). Finally, save the a TensorRT model as .pb file.
![Image of Code](https://i.imgur.com/szk1ViC.png)

The [arguments of *create_inference_graph* function](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api) are
- **input_graph_def:** input a frozen graph which is returned by  `convert_variables_to_constants` function.
- **outputs:** List of all output nodes name in the graph. The easiest way to find the list of output nodes is using TensorBoard.
- **max_batch_size:** the max size for the input batch. That means "How many images you can inference at the same time". The default value is 1.
- **max_work_space:** The maximum GPU temporary memory which the TensorRT engine can use for execution. The default value is 1GB or 1*(10**9).
- **precision_mode:** It is a data type that the optimized model can have graph and parameters stored in. The available modes are "FP32"(float32),"FP16","INT8". The default value is "FP32".


**Step 1 and 2 are in [1-main-Converting.ipynb](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/keras/1-main-Converting.ipynb)**

#### 3. Inference using TensorRT model
The inference is the stage in which a trained model is used to infer/predict the testing samples. It similar forward pass as training to predict the values. [**2-InferenceTRT.ipynb**](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/notebook/2-InferenceTRT.ipynb) is a inferencing code file. The steps are
1. Load frozen model (.pb file) which has already optimized with TensorRT
2. Import the loaded model using import_graph_def
3. Get input and output tensors
4. Write logs for TensorBoard (optional)
5. Inference using function run of TensorFlow Session.

In this case, I inference 50 times to find the average inference time.
You can repeat all these steps with the original frozen model (the frozen model without optimizing with TensorRT) for comparison.

### Comparison Result
- **Time:** Inferencing by the optimized model take less time than the original model. ![Diff Time](https://i.imgur.com/f4dAc0M.png) But if a number test predictions images is low such as 30 images, unoptimized model sometimes might be faster or equal.
- **Prediction Result:** Both predicted the same so their accuracy are the same.
![Prediction result](https://i.imgur.com/QHq2rCt.png)
- **Others metrics** such as Confusion matrix, Recall score, Precision score,  f1-score, ROC curve and AUC (area under curve): Both are the same.

### Important Reference
- Workflow: https://www.youtube.com/watch?v=AIGOSz2tFP8&list=PLkRkKTC6HZMwdtzv3PYJanRtR6ilSCZ4f
- Converting Code Guide: https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a
- Inference Code: https://github.com/ardianumam/Tensorflow-TensorRT.git
- Metrics: https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019

## Optimize Object detection model with TensorRT
The objectives of this path are optimizing the object detection models with TensorRT and comparison between the original model and optimized model. The working file is [InferenceWithTensorRT.ipynb](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/object_detection/InferenceWithTensorRT.ipynb). The workflow is almost the same as optimizing Keras model. The main differences are
- **Models** for object detection are more diversity and complexity.
- A **Dataset** is from car camera.
- Inferencing function can receive only 1 image so there must be a loop to serve and queue images to the inferencing function.

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
				+ final_config.json -> list of all objects of all images should be detected
			+ `detected_images/` -> contains images after detection
				+ images_detail.json -> list of detail infereced images include detected objects (annotations from final_config.json), filename, path, inference time, and detected numbers
			+ aadc2018_frcnn_res101_200k_kitti.pb
			+ aadc_labels_2018.pbtxt
			+ aadc_labels_2018_without_middlelane.pbtxt
			+ aadc_labels_2018_slim.pbtxt
	+ `InferenceWithTensorRT.ipynb` -> main notebook file

### Models
There are 3 models are considered.
1. `ssd_mobilenet_v1_coco_2017_11_17` is a [Single-Shot multi-box Detection (SSD)](https://medium.com/@smallfishbigsea/understand-ssd-and-implement-your-own-caa3232cd6ad) network intended to perform object detection. This model is a default model in [tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb). So, It's a starting model. You can download model [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz).
2. `faster_rcnn_resnet101_kitti_2018_01_28/` is a network for object detection. It has use cases in self-driving cars, manufacturing, security. You can download model [here](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz).
3. `aadc2018_frcnn_res101_200k_kitti/` is a given model (from professor). It is similar to faster_rcnn_kitti model but more suitable with the dataset.

### Problems and Causes
1. `ssd_mobilenet_v1_coco_2017_11_17` can inference both `test_images2` and `test_images_20181027` because it use less memory for inferencing. But this network is not suitable to detect objects in images from car camera. So, the accuracy is too low.
2. `faster_rcnn_resnet101_kitti_2018_01_28/` cannot inference because Jetson tx2 board has only 8GB memory and inferencing take a lot of memory. So, the inferencing process was automatically killed.
3. `aadc2018_frcnn_res101_200k_kitti/` is a newer version of TensorFlow so it cannot be loaded. (TensorFlow version in Jetson tx2 board is 1.9.0 which is an old version)
4. Because there is only `ssd_mobilenet_v1_coco_2017_11_17` which can inference, it is only the model that optimized with TensorRT. But, **the optimized model cannot inference.** The inferencing process was automatically killed because the process took too much memory. Although decreasing `max_batch_size to 1` and increasing `max_work_space to 5GB`, the inferencing still automatically killed. *Note that: A large max_batch_size will make large consuming memory. And, when max_work_space is exceed, a process will be automatically killed.*

### Additional
#### Precision mode in TensorRT
[Precision mode](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#precision-mode) is one of arguments have to be set in optimizing TensorRT. There are 3 available values, "FP32", "FP16", and "INT8".
Here is the result after trying optimized `ssd_mobilenet_v1_coco_2017_11_17` model with 3 different precision mode.
1. FP32: take optimizing time around 9 minutes. Can load optimized model normally.
2. FP16: take optimizing time around 12 minutes. Can load optimized model normally.
3. INT8: take optimizing time around 16 seconds!! but cannot load the optimized model.

In conclusion, precision mode INT8 take the least time and FP16 take the most time. But the optimized model of precision mode INT8 cannot be loaded. It might be because [the model need to be quantized before optimizing](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#int8-quantization).

Note that, time taking of optimizing depend on free memory space. "Much free space, less time taking".

#### Object Detection Number Check
Code for checking the inferencing result has already added. The final answer (real value) is in the `final_config.json`.
- In preparing images list for inferencing, not only collect images filename and path but also each annotations (objects in each image that should be detected) from final_config.json. The images list will look like this.
> TEST_IMAGE_LIST = [{<br>
> 'annotations' : [list of objects], <br>
> 'filename': name string,<br>
> 'path': image path<br>
> },<br>
> {<br>
> ...<br>
> }]
- `visualize_boxes_and_labels_on_image_array` function from utils is used to wrtie box of all objects in each images. It return an image with boxes so I edited to also return box number, which is the number of detected object. Now, it's easy to check the correctness of detection by comparing box number with a length of annotations.
- Add the box number and inference time into each image in the list and save it as `images_datail.json`. The `images_datail.json` will look like this.
![Images Detail](https://i.imgur.com/Z5Nx7B4.png)

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
