from Model_A import Model_A
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from Preprocess import Preprocess

class Model_B(Model_A):
    def __init__(self):
        # base class initializer syntax
        super().__init__()
        self.keep_probability = 0.25

    def create_model(self):

        x = tf.placeholder(tf.float32, shape = [None, self.row, self.col, self.channel], name = "x")

        y = tf.placeholder(tf.float32, shape = [None, self.classes], name = "y")

        keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

        #############################
        #    Convolutional layer 1  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_1_weights_tf = tf.Variable(tf.truncated_normal([3, 3, x.get_shape().as_list()[3], 48], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_1_bias_tf = tf.Variable(tf.zeros(48)) # 48 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_1_unbiased = tf.nn.conv2d(x, conv_layer_1_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_1_biased = tf.nn.bias_add(conv_layer_1_unbiased, conv_layer_1_bias_tf)
        conv_layer_1_activated = tf.nn.relu(conv_layer_1_biased)


        #############################
        #    Convolutional layer 2  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_2_weights_tf = tf.Variable(tf.truncated_normal([3, 3, conv_layer_1_activated.get_shape().as_list()[3], 48], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_2_bias_tf = tf.Variable(tf.zeros(48)) # 48 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_2_unbiased = tf.nn.conv2d(conv_layer_1_activated, conv_layer_2_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_2_biased = tf.nn.bias_add(conv_layer_2_unbiased, conv_layer_2_bias_tf)
        conv_layer_2_activated = tf.nn.relu(conv_layer_2_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows size is 2 by 2
        conv_layer_2_pooled = tf.nn.max_pool(conv_layer_2_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

        # I dropout 0.25 of the conv layer
        conv_layer_2_dropped = tf.nn.dropout(conv_layer_2_pooled, keep_prob)

        #############################
        #    Convolutional layer 3  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_3_weights_tf = tf.Variable(tf.truncated_normal([3, 3, conv_layer_2_dropped.get_shape().as_list()[3], 96], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_3_bias_tf = tf.Variable(tf.zeros(96)) # 48 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_3_unbiased = tf.nn.conv2d(conv_layer_2_dropped, conv_layer_3_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_3_biased = tf.nn.bias_add(conv_layer_3_unbiased, conv_layer_3_bias_tf)
        conv_layer_3_activated = tf.nn.relu(conv_layer_3_biased)


        #############################
        #    Convolutional layer 4  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_4_weights_tf = tf.Variable(tf.truncated_normal([3, 3, conv_layer_3_activated.get_shape().as_list()[3], 96], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_4_bias_tf = tf.Variable(tf.zeros(96)) # 48 is the number of outputs produced by the convolutional layer

        conv_layer_4_unbiased = tf.nn.conv2d(conv_layer_3_activated, conv_layer_4_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_4_biased = tf.nn.bias_add(conv_layer_4_unbiased, conv_layer_4_bias_tf)
        conv_layer_4_activated = tf.nn.relu(conv_layer_4_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows size is 2 by 2
        conv_layer_4_pooled = tf.nn.max_pool(conv_layer_4_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

        # I dropout 0.25 of the conv layer
        conv_layer_4_dropped = tf.nn.dropout(conv_layer_4_pooled, keep_prob)


        #############################
        #    Convolutional layer 5  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_5_weights_tf = tf.Variable(tf.truncated_normal([3, 3, conv_layer_4_dropped.get_shape().as_list()[3], 192], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_5_bias_tf = tf.Variable(tf.zeros(192)) # 48 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_5_unbiased = tf.nn.conv2d(conv_layer_4_dropped, conv_layer_5_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_5_biased = tf.nn.bias_add(conv_layer_5_unbiased, conv_layer_5_bias_tf)
        conv_layer_5_activated = tf.nn.relu(conv_layer_5_biased)


        #############################
        #    Convolutional layer 6  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_6_weights_tf = tf.Variable(tf.truncated_normal([3, 3, conv_layer_5_activated.get_shape().as_list()[3], 192], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_6_bias_tf = tf.Variable(tf.zeros(192)) # 48 is the number of outputs produced by the convolutional layer

        conv_layer_6_unbiased = tf.nn.conv2d(conv_layer_5_activated, conv_layer_6_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_6_biased = tf.nn.bias_add(conv_layer_6_unbiased, conv_layer_6_bias_tf)
        conv_layer_6_activated = tf.nn.relu(conv_layer_6_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows size is 2 by 2
        conv_layer_6_pooled = tf.nn.max_pool(conv_layer_6_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

        # I dropout 0.25 of the conv layer
        conv_layer_6_dropped = tf.nn.dropout(conv_layer_6_pooled, keep_prob)


        #######################
        #    flatten layer 1  #
        #######################

        flatten_layer_1 = tf.contrib.layers.flatten(conv_layer_6_dropped)

        ###############################
        #    fully connected layer 1  #
        ###############################

        fully_connected_layer_1_weights_tf = tf.Variable(tf.truncated_normal([flatten_layer_1.get_shape().as_list()[1], 512], stddev = 0.05, mean = 0.0))

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_1_bias_tf = tf.Variable(tf.zeros(512))

        fully_connected_layer_1_output = tf.add(tf.matmul(flatten_layer_1, fully_connected_layer_1_weights_tf), fully_connected_layer_1_bias_tf)

        fully_connected_layer_1_activated = tf.nn.relu(fully_connected_layer_1_output)

        fully_connected_layer_1_dropped = tf.nn.dropout(fully_connected_layer_1_activated, 0.5)

        ###############################
        #    fully connected layer 2  #
        ###############################

        fully_connected_layer_2_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_1_dropped.get_shape().as_list()[1], 256], stddev = 0.05, mean = 0.0))

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_2_bias_tf = tf.Variable(tf.zeros(256))

        fully_connected_layer_2_output = tf.add(tf.matmul(fully_connected_layer_1_dropped, fully_connected_layer_2_weights_tf), fully_connected_layer_2_bias_tf)

        fully_connected_layer_2_activated = tf.nn.relu(fully_connected_layer_2_output)

        fully_connected_layer_2_dropped = tf.nn.dropout(fully_connected_layer_2_activated, 0.5)

        ##################
        #  logits layer  #
        ##################

        output_layer_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_2_dropped.get_shape().as_list()[1], 10], stddev = 0.05, mean = 0.0))

        # one bias for each layer therefore 10 biasses
        output_layer_bias_tf = tf.Variable(tf.zeros(10))

        # logits also known as the output of the model
        logits = tf.add(tf.matmul(fully_connected_layer_2_dropped, output_layer_weights_tf), output_layer_bias_tf)

        return (x, y, keep_prob, logits)
