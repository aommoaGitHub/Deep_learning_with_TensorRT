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
The selected integration workflow is workflow with a frozen graph. It's concluded into 3 steps.
![Image of Workflow](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/doc_img/keras_trt_workflow.png)

#### 1. Convert model to frozen model
In this case, the Keras model to be converted is trained Magma model from the previous work. There is a function from TensorFlow named convert_variables_to_constants which use to freeze the model. [Sample Code.](https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a) This function can freeze both Keras and TensorFlow model and return a frozen model. **That means both of the Keras and TensorFlow model can be frozen in the same way.** You can use gflie library to save a frozen model as a .pb file that allow you to load the model in several times.

#### 2. Optimize frozen model with TensorRT
 This step use function create_inference_graph to optimize frozen model to TensorRT model. The function return TensorRT model (graph). Finally, save the a TensorRT model as .pb file.
![Image of Code](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/doc_img/optimize_trt.png)

**Step 1 and 2 are in [1-main-Converting.ipynb](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/keras/1-main-Converting.ipynb)**

#### 3. Inference using TensorRT model
Inference is the stage in which a trained model is used to infer/predict the testing samples. It similar forward pass as training to predict the values. [2-InferenceTRT.ipynb](https://gitlab.com/imla/demos/tensor_rt/blob/vittunyuta/notebook/2-InferenceTRT.ipynb) is a inferencing code file. The steps are
1. load frozen model (.pb file) which has already optimized with TensorRT
2. import the loaded model using import_graph_def
3. Get input and output tensors
4. Write logs for TensorBoard (optional)
5. Inference using function run of TensorFlow Session.

In this case, I inference 50 times to find the average inference time.
You can repeat these steps with original frozen model (the frozen model without optimizing with TensorRT) for comparison. There are some basic comparison and its result.
- Time: Inferencing with optimized model take less time than unoptimized model.
- Prediction Result: Both are the same.
- Others metrics such as Confusion matrix, Recall score, Precision score,  f1-score, ROC curve and AUC (area under curve): Both are the same.

### Important Reference
- Workflow: https://www.youtube.com/watch?v=AIGOSz2tFP8&list=PLkRkKTC6HZMwdtzv3PYJanRtR6ilSCZ4f
- Converting Code: https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a
- Inference Code: https://github.com/ardianumam/Tensorflow-TensorRT.git
- Metrics: https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019

## Optimize Object detection model with TensorRT
The workflow is the same as optimizing Keras model. The main different is **model** for object detection are more diversity and complexity.

### Directory Structure
+ `object_detection/``
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
			+ `test_images_20181027` -> contains test images
			+ `detected_images/` -> contains images after detection
			+ aadc2018_frcnn_res101_200k_kitti.pb
			+ aadc_labels_2018.pbtxt
			+ aadc_labels_2018_without_middlelane.pbtxt
			+ aadc_labels_2018_slim.pbtxt
	+ `InferenceWithTensorRT.ipynb` -> main notebook file

### Detail

### Problems
out of memory - inference faster, tensorrt
newer version of tf

### Reference
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs this link for seeing config of each model



## Daily Progress
**Date: Friday 28th June 2019**
All code for inferencing Keras Model with TensorRT is done but there are 2 bugs
1. Cannot optimize the frozen model to TensorRT graph
2. Cannot inferencing

**Date: Wednesday 3rd July 2019**
Seperate into 2 files
Converting Keras to frozen graph and then optimize to .pb files
Inferencing
Bugs of Inferencing is fixed, It is a problem related to topK parameter in DetectionOutput layer. 5000 is too large that TensorRT crushed during runtime. After I reduce to below 2500 it runs fine. Reference: https://devtalk.nvidia.com/default/topic/1037616/tensorrt/problems-with-nvidia-ssddetectionoutputplugin/

**Date: Thursday 4th July 2019**
Bugs of Creating TensorRt Inferencer is fixed. Solution is prevent create duplicate layers of frozen graph.

**Date: Friday 5th July 2019**
Finished Calculating and comparison between using original model and optimized model with TensorRT. Available measures are time and accuracy.

**Date: Monday 8th July 2019**
Fix the problems about limit of image of trt graph by increase the batch size while creation trt frozen graph. (shame mistake >///<)

**Date: Monday 15th July 2019**
Finish showing graph from pb file on Jupyter notebook. And add sklearn accuracy score and f1-score

**Date: Tuesday 16th July 2019**
Add others metrics

**Date: Thursday 18th July 2019**
Add TensorRT Optimization of Object detection

**Date: Friday 19th July 2019**
Add faster_rcnn_resnet101_kitti_2018_01_28 model and its test images. But there is an error while prediction.

**Date: Tuesday 23th July 2019**
Clean a version control such as reduce .git file by removing all large file from all commits (using command git filter-branch --tree-filter 'rm -f <path/to/file>' -- --all) and cleanup unnecessary files (using command git gc --aggressive --prune=now), make sure a repository is correct and up to date.

**Date: Wednesday 24th July 2019**
Add aadc2018_frcnn_res101_200k_kitti model and its test images. But the model cannot import. It may be because the model is newer version of Tensorflow than Tensorflow version on the Jetson board.

**Date: Thursday 25th July 2019**
Add code for checking annotations(found items) of each image, which is used for calculation accuracy. A cause of the problem while prediction is "Out of memory". A prediction process use too much memory so it is killed.


## Moreover Source (not done yet)
### Face detection - Hall of face
- https://github.com/the-house-of-black-and-white/hall-of-faces
Area under the ROC curve show how good the model of detection.
(mAP measurements) https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

### Video - Object Detection
- YOLO Realtim: https://github.com/ardianumam/Tensorflow-TensorRT.git
