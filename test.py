# test.py
#
# This file is modified form of :
# https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_2_Classification_Walk-through/blob/master/test.py

import os
import tensorflow as tf
import numpy as np
import cv2
from model import get_model
from data_preprocess import img_size,get_data,get_test_data,TEST_IMAGES_DIR

# module-level variables ##############################################################################################
model_path = "model/bike_classifier"
num_classes = 2

# where to save summary logs for TensorBoard
TENSORBOARD_DIR = os.getcwd() + '/tboard/test'

# save output results
RESULTS_DIR = os.getcwd() + '/results/'

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

#######################################################################################################################
def main():
  print("starting test . . .")

  if not checkIfNecessaryPathsAndFilesExist():
      return
  # end if

  #get the class label strings
  _,_,_, classifications = get_data()

  # Build the graph for the cnn
  x = tf.placeholder(tf.float32, [None, img_size,img_size,3], name='in')
  y_ = tf.placeholder(tf.int64, [None],name='y_')
  y_conv, keep_prob = get_model(x,img_size,num_classes)

  # 'Saver' op to save and restore all the variables
  saver = tf.train.Saver()

  # Softmax loss 
  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  # accuracy 
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_, name="final")
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
        # for each file in the test images directory . . .
        for fileName in os.listdir(TEST_IMAGES_DIR):
            # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            # end if

            # show the file name on std out
            print(fileName)

            # get the file name and full path of the current image file
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            # attempt to open the image with OpenCV
            img = cv2.imread(imageFileWithPath)
            openCVImage = cv2.resize(img, (img_size, img_size)).astype(np.float32)/255.0

            # if we were not able to successfully open the image, continue with the next iteration of the for loop
            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue
            # end if

            # get the final tensor from the graph
            finalTensor = sess.graph.get_tensor_by_name('y_:0')

            # convert the OpenCV image (numpy array) to a TensorFlow image
            tfImage = np.array(openCVImage)[:, :, 0:3]
            tfImage = np.expand_dims(tfImage,axis=0)
            
            # run the network to get the predictions
            predictions = sess.run(tf.nn.softmax(y_conv), {x: tfImage,keep_prob:1.0})


            # sort predictions from most confidence to least confidence
            sortedPredictions = np.flip(predictions[0].argsort(),0)#=[-len(predictions[0]):][::-1]

            print("---------------------------------------")

            # keep track of if we're going through the next for loop for the first time so we can show more info about
            # the first prediction, which is the most likely prediction (they were sorted descending above)
            onMostLikelyPrediction = True
            # for each prediction . . .
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]
                # end if

                # get confidence, then get confidence rounded to 2 places after the decimal
                confidence = predictions[0][prediction]

                # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
                if onMostLikelyPrediction:
                    # get the score as a %
                    scoreAsAPercent = confidence * 100.0
                    # show the result to std out
                    print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    print(predictions)
                    # write the result on the image
                    writeResultOnImage(img, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    # finally we can show the OpenCV image
                    cv2.imshow(fileName, img)
                    # cv2.imwrite(RESULTS_DIR+ fileName,img)
                    # mark that we've show the most likely prediction at this point so the additional information in
                    # this if statement does not show again for this image
                    onMostLikelyPrediction = False
                # end if

                # for any prediction, show the confidence as a ratio to five decimal places
                print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")
            # end for

            # pause until a key is pressed so the user can see the current image (shown above) and the prediction info
            cv2.waitKey()
            # after a key is pressed, close the current window to prep for the next time around
            cv2.destroyAllWindows()
        # end for
        # end with

        # write the graph to file so we can view with TensorBoard
        tfFileWriter = tf.summary.FileWriter(TENSORBOARD_DIR)
        tfFileWriter.add_graph(sess.graph)
        tfFileWriter.close()

# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return False
    # end if

    # if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
    #     print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
    #     return False
    # # end if

    # if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
    #     print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
    #     return False
    # # end if

    return True
# end function

#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 1.0
    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
