'''
Author: Md. Tanvir Islam
'''

import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from Preprocess import Preprocess

class Model_A(object):

    def __init__(self):
        data = Preprocess()

        data.load_data()

        data.preprocess_images()

        data.one_hot_encode_labels()

        self.train_x, self.train_y = data.get_training_data()

        self.test_x, self.test_y = data.get_testing_data()

        _, _, self.row, self.col, self.channel, self.classes = data.metadata()

        self.batch_size = 256

        self.keep_probability = 0.4

        self.name = "Model-A"

    def create_model(self):

        x = tf.placeholder(tf.float32, shape = [None, self.row, self.col, self.channel], name = "inputs")

        y = tf.placeholder(tf.float32, shape = [None, self.classes], name = "outputs")

        keep_prob = tf.placeholder(tf.float32, name = "keep-prob")

        #############################
        #    Convolutional layer 0  #
        #############################
        num_filters_l0 = 12
        # tf variable for weights in the convolutional layer
        #                                                               3 since our image has 3 channels
        conv_layer_0_weights_tf = tf.Variable(tf.truncated_normal([3, 3, self.channel, num_filters_l0], stddev=0.05, mean=0.0), name = "conv-layer-1-weights") # 24 is the number of filteres being applied
        # tf variable for the bias weights
        conv_layer_0_bias_tf = tf.Variable(tf.zeros(num_filters_l0), name = "conv1-layer-1-biases") # 24 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_0_unbiased = tf.nn.conv2d(x, conv_layer_0_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_0_biased = tf.nn.bias_add(conv_layer_0_unbiased, conv_layer_0_bias_tf)
        conv_layer_0_activated = tf.nn.relu(conv_layer_0_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows and the pool size is 2 by 2
        conv_layer_0_pooled = tf.nn.max_pool(conv_layer_0_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')


        #############################
        #    Convolutional layer 1  #
        #############################

        num_filters_l1 = 24
        # tf variable for weights in the convolutional layer
        #                                                               3 since our image has 3 channels
        conv_layer_1_weights_tf = tf.Variable(tf.truncated_normal([3, 3, num_filters_l0, num_filters_l1], stddev=0.05, mean=0.0), name = "conv-layer-2-weights") # 24 is the number of filteres being applied
        # tf variable for the bias weights
        conv_layer_1_bias_tf = tf.Variable(tf.zeros(num_filters_l1), name = "conv-layer-2-biases") # 24 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_1_unbiased = tf.nn.conv2d(conv_layer_0_pooled, conv_layer_1_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_1_biased = tf.nn.bias_add(conv_layer_1_unbiased, conv_layer_1_bias_tf)
        conv_layer_1_activated = tf.nn.relu(conv_layer_1_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows and the pool size is 2 by 2
        conv_layer_1_pooled = tf.nn.max_pool(conv_layer_1_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')


        #############################
        #    Convolutional layer 2  #
        #############################

        num_filters_l2 = 48
        # tf variable for weights in the convolutional layer
        #                                                               x.get_shape().as_list()[3] should be 24 since the filtered image spitted out by the previous convolutional layer should have 24 images all with 1 channel or 1 image with 24 channel
        conv_layer_2_weights_tf = tf.Variable(tf.truncated_normal([3, 3, num_filters_l1, num_filters_l2], stddev=0.05, mean=0.0), name = "conv-layer-3-weights") # 48 is the number of filters being applied
        # tf variable for the bias weights
        conv_layer_2_bias_tf = tf.Variable(tf.zeros(num_filters_l2), name = "conv-layer-3-biases") # 48 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_2_unbiased = tf.nn.conv2d(conv_layer_1_pooled, conv_layer_2_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_2_biased = tf.nn.bias_add(conv_layer_2_unbiased, conv_layer_2_bias_tf)
        conv_layer_2_activated = tf.nn.relu(conv_layer_2_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows and the pool size is 2 by 2
        conv_layer_2_pooled = tf.nn.max_pool(conv_layer_2_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

        #############################
        #    Convolutional layer 3  #
        #############################

        num_filters_l3 = 96
        # tf variable for weights in the convolutional layer
        conv_layer_3_weights_tf = tf.Variable(tf.truncated_normal([3, 3, num_filters_l2, num_filters_l3], stddev=0.05, mean=0.0), name = "conv-layer-4-weights")
        # tf variable for the bias weights
        conv_layer_3_bias_tf = tf.Variable(tf.zeros(num_filters_l3), name = "conv-layer-4-biases") # 48 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_3_unbiased = tf.nn.conv2d(conv_layer_2_pooled, conv_layer_3_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_3_biased = tf.nn.bias_add(conv_layer_3_unbiased, conv_layer_3_bias_tf)
        conv_layer_3_activated = tf.nn.relu(conv_layer_3_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows and the pool size is 2 by 2
        conv_layer_3_pooled = tf.nn.max_pool(conv_layer_3_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')


        #######################
        #    flatten layer 1  #
        #######################

        flatten_layer_1 = tf.contrib.layers.flatten(conv_layer_3_pooled)

        fc_neurons = 512
        ###############################
        #    fully connected layer 1  #
        ###############################

        fully_connected_layer_1_weights_tf = tf.Variable(tf.truncated_normal([flatten_layer_1.get_shape().as_list()[1], fc_neurons], stddev = 0.05, mean = 0.0), name = "fully-connected-layer-1-weights")

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_1_bias_tf = tf.Variable(tf.zeros(512), name = "fully-connected-layer-1-biases")

        fully_connected_layer_1_output = tf.add(tf.matmul(flatten_layer_1, fully_connected_layer_1_weights_tf), fully_connected_layer_1_bias_tf)

        fully_connected_layer_1_activated = tf.nn.relu(fully_connected_layer_1_output)

        fully_connected_layer_1_dropped = tf.nn.dropout(fully_connected_layer_1_activated, keep_prob)

        ###############################
        #    fully connected layer 2  #
        ###############################

        fully_connected_layer_2_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_1_dropped.get_shape().as_list()[1], fc_neurons], stddev = 0.05, mean = 0.0), name = "fully-connected-layer-2-weights")

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_2_bias_tf = tf.Variable(tf.zeros(512), name = "fully-connected-layer-2-biases")

        fully_connected_layer_2_output = tf.add(tf.matmul(fully_connected_layer_1_dropped, fully_connected_layer_2_weights_tf), fully_connected_layer_2_bias_tf)

        fully_connected_layer_2_activated = tf.nn.relu(fully_connected_layer_2_output)

        fully_connected_layer_2_dropped = tf.nn.dropout(fully_connected_layer_2_activated, keep_prob)

        ##################
        #  logits layer  #
        ##################

        output_layer_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_2_dropped.get_shape().as_list()[1], self.classes], stddev = 0.05, mean = 0.0), name = "output-layer-weights")

        # one bias for each layer therefore 10 biasses
        output_layer_bias_tf = tf.Variable(tf.zeros(self.classes), name = "output-layer-biases")

        # logits also known as the output of the model
        logits = tf.add(tf.matmul(fully_connected_layer_2_dropped, output_layer_weights_tf), output_layer_bias_tf)

        print("Model-A created...")

        return (x, y, keep_prob, logits)


    def train(self):
        epochs = 15
        average_accuracy = 0

        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()

        x, y, keep_prob, logits = self.create_model()

        with tf.name_scope("cost"):
            # loss of the neural network
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))

            # optimizer will take in loss and minimize it using gradient descent
            optimizer = tf.train.AdamOptimizer().minimize(loss)

            tf.summary.scalar("cost", loss)

        with tf.name_scope("accuracy"):
            # Accuracy of the model
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

            tf.summary.scalar("accuracy", accuracy)

        # TRAINING WILL BEGIN NOW
        print("Initiating tensorflow session...")
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./logs/" + self.name, sess.graph)
            merged = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())

            begin = 0
            finish = 256

            for i in range(epochs):
                print("On epoch no.{}".format(i + 1))

                training_batch = zip(range(0, len(self.train_x), 128), range(self.batch_size, len(self.train_x) + 1, 128))

                for start, end in tqdm(training_batch):
                    sess.run(optimizer, feed_dict = {x: self.train_x[start:end], y: self.train_y[start:end], keep_prob: self.keep_probability})


                # For printing purposes
                l = sess.run(loss, feed_dict = {x: self.train_x[start:end], y: self.train_y[start:end], keep_prob: 1.})

                summary, valid_acc = sess.run([merged, accuracy], feed_dict = {x: self.test_x[begin:finish], y: self.test_y[begin:finish], keep_prob: 1.})

                writer.add_summary(summary, i)

                print("Loss: {:>10.4f} Accuracy: {}".format(l, valid_acc))

                average_accuracy += valid_acc

                begin = finish
                finish += 256

                # we break the loop if we run out of testing data
                if (begin > len(self.test_x)):
                    break

            average_accuracy /= epochs

            print("Average accuracy of the network: {}".format(average_accuracy))
