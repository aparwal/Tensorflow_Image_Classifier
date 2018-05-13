#!/bin/env python3
# model.py
# Genetrate a basic CNN model in tensorflow
# Inspired from examples provided in tensorflow repo
# __author__ = Anand Parwal   

import tensorflow as tf

def get_model(x,img_size,num_classes):
  """get_model builds the graph for a cnn for classifying bikes.

  Args:
    x: an input tensor with the dimensions (N_examples, img_size*img_size*3), where 
    img_size: Length of a side of resized input image in pixels.
    num_classes: Number of classes in the output

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, num_classes), with values
    equal to the logits of classifying the digit into one of num_classes classes. 
    keep_prob is a scalar placeholder for the probability of dropout.
  """

  # Reshape to use within a convolutional neural net. 
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, img_size, img_size, 3])
    tf.summary.image('input', x_image, num_classes)

  # First convolutional layer - maps one rbg image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1 )
    variable_summaries(b_conv1)
    variable_summaries(W_conv1)

  # first pooling layer.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
  
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    variable_summaries(b_conv2)
    variable_summaries(W_conv2)

  # second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our image
  # is down to (32/4)x(32/4)x64 feature maps -- maps this to 256 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([(img_size//16) * (img_size//16) * 64, 256])
    b_fc1 = bias_variable([256])
    variable_summaries(b_fc1)
    variable_summaries(W_fc1)


    h_pool2_flat = tf.reshape(h_pool2, [-1, (img_size//16) * (img_size//16) * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram('activations', h_fc1)

  # Dropout layer
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 256 features to 2 classes
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([256, num_classes])
    b_fc2 = bias_variable([num_classes])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 4X."""
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
