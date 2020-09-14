import cv2
import numpy as np
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_FROZEN_GRAPH = 'C:/Users/upadh/Desktop/pubg_v3/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/upadh/Desktop/pubg_v3/labelmap.pbtxt'
NUM_CLASSES = 4

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

arr=[]
cap=cv2.VideoCapture('C:/Users/upadh/Videos/vid17.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fp=int(cap.get(5))
print(fp)
out = cv2.VideoWriter('C:/Users/upadh/Videos/od/ObjectDetection17.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 21, (frame_width,frame_height))

# # Detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      # Get raw pixels from the screen, save it to a Numpy array
      _,img=cap.read()
      if not _:
        cv2.destroyAllWindows()
        break
      image_np = np.array(img)
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
      out.write(image_np)
      cv2.imshow("win", cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
      

      if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
for i in arr:
  cv2.imshow('win',cv2.cvtColor(i, cv2.COLOR_BGR2RGB))