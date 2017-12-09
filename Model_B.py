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
        self.name = "Model-B"

    def create_model(self):

        x = tf.placeholder(tf.float32, shape = [None, self.row, self.col, self.channel], name = "x")

        y = tf.placeholder(tf.float32, shape = [None, self.classes], name = "y")

        keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

        #############################
        #    Convolutional layer 1  #
        #############################

        num_filters_l1 = 24
        # tf variable for weights in the convolutional layer
        conv_layer_1_weights_tf = tf.Variable(tf.truncated_normal([2, 2, self.channel, num_filters_l1], stddev=0.01, mean=0.0)) # 24 is the number of filters being applied
        # tf variable for the bias weights
        conv_layer_1_bias_tf = tf.Variable(tf.zeros(num_filters_l1))

        # x is the input layer
        conv_layer_1_unbiased = tf.nn.conv2d(x, conv_layer_1_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_1_biased = tf.nn.bias_add(conv_layer_1_unbiased, conv_layer_1_bias_tf)
        conv_layer_1_activated = tf.nn.relu(conv_layer_1_biased)
        conv_layer_1_pooled = tf.nn.max_pool(conv_layer_1_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

        #############################
        #    Convolutional layer 2  #
        #############################

        num_filters_l2 = 24

        # tf variable for weights in the convolutional layer
        conv_layer_2_weights_tf = tf.Variable(tf.truncated_normal([3, 3, num_filters_l1, num_filters_l2], stddev=0.01, mean=0.0)) # 24 is the number of filters being applied
        # tf variable for the bias weights
        conv_layer_2_bias_tf = tf.Variable(tf.zeros(num_filters_l2))

        # x is the input layer
        conv_layer_2_unbiased = tf.nn.conv2d(conv_layer_1_pooled, conv_layer_2_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_2_biased = tf.nn.bias_add(conv_layer_2_unbiased, conv_layer_2_bias_tf)
        conv_layer_2_activated = tf.nn.relu(conv_layer_2_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows size is 2 by 2
        conv_layer_2_pooled = tf.nn.max_pool(conv_layer_2_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')


        #######################
        #    flatten layer 1  #
        #######################

        flatten_layer_1 = tf.contrib.layers.flatten(conv_layer_2_pooled)

        ###############################
        #    fully connected layer 1  #
        ###############################

        fc_neurons = 512

        fully_connected_layer_1_weights_tf = tf.Variable(tf.truncated_normal([num_filters_l2, fc_neurons], stddev = 0.01, mean = 0.0))

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_1_bias_tf = tf.Variable(tf.zeros(fc_neurons))

        fully_connected_layer_1_output = tf.add(tf.matmul(flatten_layer_1, fully_connected_layer_1_weights_tf), fully_connected_layer_1_bias_tf)

        fully_connected_layer_1_activated = tf.nn.relu(fully_connected_layer_1_output)

        fully_connected_layer_1_dropped = tf.nn.dropout(fully_connected_layer_1_activated, keep_prob)

        ###############################
        #    fully connected layer 2  #
        ###############################

        fc_neurons = 256

        fully_connected_layer_2_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_1_dropped.get_shape().as_list()[1], fc_neurons], stddev = 0.01, mean = 0.0))

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_2_bias_tf = tf.Variable(tf.zeros(fc_neurons))

        fully_connected_layer_2_output = tf.add(tf.matmul(fully_connected_layer_1_dropped, fully_connected_layer_2_weights_tf), fully_connected_layer_2_bias_tf)

        fully_connected_layer_2_activated = tf.nn.relu(fully_connected_layer_2_output)

        fully_connected_layer_2_dropped = tf.nn.dropout(fully_connected_layer_2_activated, keep_prob)

        ##################
        #  logits layer  #
        ##################

        output_layer_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_2_dropped.get_shape().as_list()[1], self.classes], stddev = 0.01, mean = 0.0))

        # one bias for each layer therefore 10 biasses
        output_layer_bias_tf = tf.Variable(tf.zeros(self.classes))

        # logits also known as the output of the model
        logits = tf.add(tf.matmul(fully_connected_layer_2_dropped, output_layer_weights_tf), output_layer_bias_tf)

        print("Model-B created...")

        return (x, y, keep_prob, logits)
