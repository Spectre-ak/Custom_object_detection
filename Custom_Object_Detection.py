# Custom Object Detection on Google Colab using Tensorflow Object Detection API 

## Installing the dependencies

!apt-get install protobuf-compiler python-pil python-lxml python-tk
!pip install Cython tf_slim
!git clone https://github.com/tensorflow/models.git
%cd /content/models/research
!protoc object_detection/protos/*.proto --python_out=.
%set_env PYTHONPATH=/content/models/research:/content/models/research/slim

!python setup.py build
!python setup.py install

import os
os.environ['PYTHONPATH'] += ":/content/models"
import sys
sys.path.append("/content/models")

## Using faster_rcnn_inception_v2_coco_2018_01_28 model which trained over coco dataset and which will be trained again on our custom dataset

import tensorflow as tf
U='http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'
zip_dir = tf.keras.utils.get_file('faster.tgz', origin=U, extract=True)
print(zip_dir)
import tarfile
mt=tarfile.open(zip_dir)
mt.extractall('./')
mt.close()

# train.py uses tensorflow 1.x and to work on gpu properly tensorflow must be 1.15

!pip install tensorflow-gpu==1.15

%cd /content/models/research
%set_env PYTHONPATH=/content/models/research:/content/models/research/slim
!python object_detection/builders/model_builder_test.py

# Mounting the google drive so that the no training data will be lost 

from google.colab import drive
drive.mount('/content/drive')

# image datset

import zipfile
with zipfile.ZipFile('/content/drive/My Drive/pubg_v3/images_v3.zip', 'r') as zip_ref:
    zip_ref.extractall('./content')

# this is done to convert the xml files of images into csv file so that .record file can be generated

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
def main():
    for folder in ['images_v3']:
        image_path = os.path.join(os.getcwd(), ('content/' + folder))
        print(image_path)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('/content/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')
main()

 # tf_record.py is use to generate .record files for the data using the xml files and images

!python tf_record.py --csv_input=/content/models/research/content/images_v3_labels.csv --image_dir=/content/models/research/content/images_v3 --output_path=/content/tf_record/train.record

# train.py is use to train the model using the config file and training dir stores all checkpoints during the training process. This model will be trained over the images_v3(custom) dataset for 50,000 steps, however the number of steps can be changed in the config file.

%%capture
!python train.py --logtostderr --train_dir="/content/drive/My Drive/pubg_v3/training_dir" --pipeline_config_path=faster_rcnn_inception_v2_coco.config

# After training the training_dir stores all the checkpoints upto the last step and any of those files can be used to extract the inference graph using the export_inference_graph.py file. 

!python export_inference_graph.py --input_type image_tensor --pipeline_config_path faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix "/content/drive/My Drive/pubg_v3/training_dir/model.ckpt-50000" --output_directory /content/inference_graph

# Now that we have the inference_graph.pb we can make the prediction over images and videos. Here we can use visualization_utils from object_detection api to draw the boxes on images.

 
import time
import cv2
#import mss
import numpy as np
import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
from google.colab.patches import cv2_imshow
# ## Env setup
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
 
 
# # Model preparation 
PATH_TO_FROZEN_GRAPH = '/content/inference_graph/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/content/labelmap.pbtxt'
NUM_CLASSES = 4
 
 
# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
 
# # Detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      # Get raw pixels from the screen, save it to a Numpy array
      img=Image.open('/content/fpp-mode-for-pubg-mobile-might-be-right-around-corner.w1456.jpg')
      img=img.resize((1024,600))
      image_np = np.array(img)
      # To get real color we do this:
      #image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Visualization of the results of a detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2)
      # Show image with detection
      image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
      cv2_imshow(image_np)
      break



