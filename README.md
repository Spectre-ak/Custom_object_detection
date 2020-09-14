# Custom_object_detection
Custom object detection on Google Colab using TensorFlow object detection API.
Here you'll find how to train your custom object detection model on Google Colab. And scripts are for Colab Jupyter notebook so these scripts will not work on your local system without making some changes(paths).

Dataset: For this custom object detection, I am using First Person Shooter games for deteting Enemy and Enemy head and Vehicle. Total of I have 2000 images for training.
LabelImg tool for making the xml files form images is [here](https://tzutalin.github.io/labelImg/)
Using the above tool we get the xml file for each image which contains the bounding boxe for the object to be detected.

After making the xml files of all images .csv file is created [here](https://github.com/Spectre-ak/Custom_object_detection/blob/master/Custom_Object_Detection.py).

#### I've used tensorflow-gpu 1.15 for custom object detection on colab.
Object detection model zoo for Tensorflow 1 is [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and Tensorflow 2 is [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

Here I am using [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) model of Tensorflow 1.








