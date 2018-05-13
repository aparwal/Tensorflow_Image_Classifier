#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH -o train.out
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:6
#SBATCH --mem=126000

##################################
# train.py
# Main file used to train the cnn on data
# __author__ = Anand Parwal 
# inspired by:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py
###################################

import tensorflow as tf
from numpy import hstack,vstack
# from the current package
from model import get_model
from data_preprocess import img_size,get_data,get_test_data


# Training Parameters
learning_rate = 0.001
num_epochs = 50
batch_size = 1
display_step = 10

# Network Parameters
# img_size =  data_preproces.img_size 
num_classes = 2
dropout = 0.75 # Dropout, probability to keep units

# Directory for tensorboard
graph_location = "tboard/train"
# path to save the model after training
model_path = "model/bike_classifier"

def main():
  ''' Main function used to train the cnn'''

  # Get the data
  train_x, train_y, _, _ = get_data()

  # Split into train and validation
  split_size = int(train_x.shape[0]*0.1)
  train_x, val_x = vstack([train_x[:split_size*4],train_x[split_size*5:split_size*9]]), vstack([train_x[split_size*4:split_size*5],train_x[split_size*9:]])
  train_y, val_y = hstack([train_y[:split_size*4],train_y[split_size*5:split_size*9]]), hstack([train_y[split_size*4:split_size*5],train_y[split_size*9:]])

  # Create the model
  x = tf.placeholder(tf.float32, [None, img_size,img_size,3])
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the cnn
  y_conv, keep_prob = get_model(x,img_size,num_classes)

  print('***********************************')

  # Softmax loss 
  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  # adam optimizer
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

  # accuracy 
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # tensorboard logging
  tf.summary.scalar('loss', cross_entropy)
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  
  # Tensorboard writing
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  # 'Saver' op to save and restore all the variables
  saver = tf.train.Saver(max_to_keep = 1)

  # Actual training
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train for num_epochs steps
    for i in range(num_epochs):
      # batch = mnist.train.next_batch(50)
      _,summary=sess.run([train_step,merged], feed_dict={x: train_x, y_: train_y, keep_prob: dropout})
      train_writer.add_summary(summary,i) 
      if i % display_step == 0:
        train_accuracy = accuracy.eval( feed_dict={ x: train_x, y_: train_y, keep_prob: 1})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        val_accuracy = accuracy.eval( feed_dict={ x: val_x, y_: val_y, keep_prob: 1})
        print('step %d, validation accuracy %g' % (i, val_accuracy))
        

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path) 

    # test_x, test_y, _ = get_test_data()
    # print('test accuracy %g' % accuracy.eval( feed_dict={x: test_x, y_: test_y, keep_prob: 1.0}))

if __name__ == '__main__':
  main()
