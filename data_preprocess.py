#!/bin/env python3
# data_preprocess.py
# data processing pipeline for train.py
# __author__ = Anand Parwal 

import os
import numpy as np
import cv2

# path to folders of labeled images
TRAINING_IMAGES_DIR = os.getcwd() + '/data/train'

TEST_IMAGES_DIR = os.getcwd() + "/data/test/"

# where to save summary logs for TensorBoard
TENSORBOARD_DIR = os.getcwd() + '/' + 'tensorboard_logs'

# edge length of the final square image fed to the network
img_size = 128

# whether to randomly flip half of the training images horizontally
FLIP_LEFT_RIGHT = False

# a percentage determining how much of a margin to randomly crop off the training images
RANDOM_CROP = 0

# a percentage determining how much to randomly scale up the size of the training images by
RANDOM_SCALE = 0

# a percentage determining how much to randomly multiply the training image input pixels up or down by
RANDOM_BRIGHTNESS = 0


def get_data(path=TRAINING_IMAGES_DIR):
  '''reads the data from the given folder
  
  Args:
    path: directory to the data whose subdirectories are labels
  Returns:
    np.array(images): numpy array of all the images 
        of size (number of total images,img_size,img_size,3)
    np.array(y): numpy array of image labels 
        of size (number of total images)
    file_names: list of string image filenames
    labels: list of string labels
    '''

  images=[]
  file_names=[]
  y=[]

  # Names of the subdirectories
  labels = os.listdir(path)
  # For order consistancy 
  labels.sort()

  # read the labels
  for label in labels:
    
    # Read every image in a subdiretory 
    for image_filename in os.listdir(os.path.join(path,label)):

      # index of image labels
      # id = np.zeros(len(labels))
      # id[labels.index(label)] = 1
      id = labels.index(label)
      y.append(id)

      #open, resize and save image in the list
      filepath= os.path.join(path, label, image_filename)
      file_names.append(filepath)
      img = cv2.imread(filepath)
      img = cv2.resize(img, (img_size, img_size)).astype(np.float32)/255.0
      images.append(img)

  return np.array(images),np.array(y),file_names,labels

def get_test_data(path=TEST_IMAGES_DIR):
  ''' utility function to get test images
  '''
  return get_data(path)


if __name__ == '__main__':
    x,y,_,_=get_data()
    print(x.shape,y.shape)