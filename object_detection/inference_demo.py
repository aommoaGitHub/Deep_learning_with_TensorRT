import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as PILImage

# For import utils.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util
from utils import visualization_utils as vis_util

import pprint
import time
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

sub_dir = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
print(sub_dir)

# What model to download.
MODEL_NAME = './models/faster_rcnn_resnet101_kitti_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
OUTPUT_NODES = ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'kitti_label_map.pbtxt')
print(PATH_TO_LABELS)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
print(PATH_TO_FROZEN_GRAPH)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
    with gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def imshow(img):
    import cv2
    import IPython
    _,ret = cv2.imencode('.jpg',img)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)

PATH_TO_TEST_IMAGES_DIR = './models/aadc2018_frcnn_res101_200k_kitti/test_images_20181027'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, file) for file in os.listdir(PATH_TO_TEST_IMAGES_DIR) if file.endswith(('jpg', 'png'))  ]
pprint.pprint(TEST_IMAGE_PATHS)

FINAL_CONFIG_FILE = PATH_TO_TEST_IMAGES_DIR + "/final_config.json"
print(FINAL_CONFIG_FILE)

tf.reset_default_graph()

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        print('Read frozen model')
        tf_graph = read_pb_graph(PATH_TO_FROZEN_GRAPH)
#         show_graph(graph_def=tf_graph)
        tf.import_graph_def(tf_graph, name='')
    
        # write to tensorboard (check tensorboard for each op names)
        #writer = tf.summary.FileWriter('./logs/'+sub_dir)
        #writer.add_graph(sess.graph)
        #writer.flush()
        #writer.close()
        #print("\nWrite logs {} success\n".format(sub_dir))
        
        #frozen_graph = tf.graph_util.convert_variables_to_constants(
        #    sess, # session
        #    tf.get_default_graph().as_graph_def(),# graph+weight from the session
        #    output_node_names=OUTPUT_NODES)

OUTPUTS = ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in OUTPUTS:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
#             print('tensor dict')
#             pprint.pprint(tensor_dict)
            
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            print("Run Inference")
            print("image size:",image.shape)
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image}) ## Boom here
            print("Finish Inference")
            
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

total_time = 0

for idx, image_path in enumerate(TEST_IMAGE_PATHS):
    print(image_path)
    image = PILImage.open(image_path)
    image_np = load_image_into_numpy_array(image)
#     print(image_np.shape)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    # Actual detection.
    t1 = time.time()
    print("--")
    output_dict = run_inference_for_single_image(image_np_expanded, graph)
    print(output_dict)    
    print("---")
    t2 = time.time()
    delta_time = t2 - t1
    total_time += delta_time
    
#     print(output_dict)
    
    # Visualization of the results of a detection.
    image_np, box_num = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    print("box num:",box_num)
    print("----")
    # Saving image
    img_name = image_path.replace('test_images_20181027','detected_images')
    # for demo images
    if img_name == image_path:
        img_name = "./test_images2/%d.jpg"%idx
    print(idx,'- Saving images:', img_name, ",size:",image_np.shape, "used time:", delta_time)
    im = PILImage.fromarray(image_np, mode="RGB")
#     im.save("detected_images/"+img_name)
    im.save(img_name)
#     cv2.imwrite("./detected_images/"+idx+".jpg",image_np)
#     imshow(image_np)
    
    if idx==2 :    
        break
    
print("Total time:",total_time)
