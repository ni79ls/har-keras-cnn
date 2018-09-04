# har-keras-cnn
Human Activity Recognition (HAR) with 1D Convolutional Neural Network in Python and Keras

A CNN works well for identifying simple patterns within your data which will then be used to form more complex patterns within higher layers. A 1D CNN is very effective when you expect to derive interesting features from shorter (fixed-length) segments of the overall data set and where the location of the feature within the segment is not of high relevance. This applies well to the analysis of time sequences of sensor data (such as gyroscope or accelerometer data).

In this example we will train a 1D convolutional neural network (1D CNN) to recognize the type of movement (Walking, Running, Jogging, etc.) based on a given set of accelerometer data from a mobile device carried around a person's waist.

We will use the WISDM data set (Activity Prediction) for this tutorial: http://www.cis.fordham.edu/wisdm/dataset.php
