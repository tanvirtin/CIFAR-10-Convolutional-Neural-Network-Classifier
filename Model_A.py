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

        self.keep_probability = 0.5

    def create_model(self):

        x = tf.placeholder(tf.float32, shape = [None, self.row, self.col, self.channel], name = "x")

        y = tf.placeholder(tf.float32, shape = [None, self.classes], name = "y")

        keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

        #############################
        #    Convolutional layer 1  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_1_weights_tf = tf.Variable(tf.truncated_normal([3, 3, x.get_shape().as_list()[3], 24], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_1_bias_tf = tf.Variable(tf.zeros(24)) # 24 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_1_unbiased = tf.nn.conv2d(x, conv_layer_1_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_1_biased = tf.nn.bias_add(conv_layer_1_unbiased, conv_layer_1_bias_tf)
        conv_layer_1_activated = tf.nn.relu(conv_layer_1_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows and the pool size is 2 by 2
        conv_layer_1_pooled = tf.nn.max_pool(conv_layer_1_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')


        #############################
        #    Convolutional layer 2  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_2_weights_tf = tf.Variable(tf.truncated_normal([3, 3, conv_layer_1_pooled.get_shape().as_list()[3], 48], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_2_bias_tf = tf.Variable(tf.zeros(48)) # 48 is the number of outputs produced by the convolutional layer

        # x is the input layer
        conv_layer_2_unbiased = tf.nn.conv2d(conv_layer_1_pooled, conv_layer_2_weights_tf, strides=[1, 1, 1, 1], padding = 'SAME')
        conv_layer_2_biased = tf.nn.bias_add(conv_layer_2_unbiased, conv_layer_2_bias_tf)
        conv_layer_2_activated = tf.nn.relu(conv_layer_2_biased)

        # now I pool the convolutional layer by taking out the max value from specific windows and the pool size is 2 by 2
        conv_layer_2_pooled = tf.nn.max_pool(conv_layer_2_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

        #############################
        #    Convolutional layer 3  #
        #############################

        # tf variable for weights in the convolutional layer
        conv_layer_3_weights_tf = tf.Variable(tf.truncated_normal([3, 3, conv_layer_2_pooled.get_shape().as_list()[3], 96], stddev=0.05, mean=0.0))  # 24 is the number of outputs produced by the convolutional layer

        # tf variable for the bias weights
        conv_layer_3_bias_tf = tf.Variable(tf.zeros(96)) # 48 is the number of outputs produced by the convolutional layer

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

        ###############################
        #    fully connected layer 1  #
        ###############################

        fully_connected_layer_1_weights_tf = tf.Variable(tf.truncated_normal([flatten_layer_1.get_shape().as_list()[1], 512], stddev = 0.05, mean = 0.0))

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_1_bias_tf = tf.Variable(tf.zeros(512))

        fully_connected_layer_1_output = tf.add(tf.matmul(flatten_layer_1, fully_connected_layer_1_weights_tf), fully_connected_layer_1_bias_tf)

        fully_connected_layer_1_activated = tf.nn.relu(fully_connected_layer_1_output)

        fully_connected_layer_1_dropped = tf.nn.dropout(fully_connected_layer_1_activated, keep_prob)

        ###############################
        #    fully connected layer 2  #
        ###############################

        fully_connected_layer_2_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_1_dropped.get_shape().as_list()[1], 512], stddev = 0.05, mean = 0.0))

        # one bias for each layer therefore 512 biasses
        fully_connected_layer_2_bias_tf = tf.Variable(tf.zeros(512))

        fully_connected_layer_2_output = tf.add(tf.matmul(fully_connected_layer_1_dropped, fully_connected_layer_2_weights_tf), fully_connected_layer_2_bias_tf)

        fully_connected_layer_2_activated = tf.nn.relu(fully_connected_layer_2_output)

        fully_connected_layer_2_dropped = tf.nn.dropout(fully_connected_layer_2_activated, keep_prob)

        ##################
        #  logits layer  #
        ##################

        output_layer_weights_tf = tf.Variable(tf.truncated_normal([fully_connected_layer_2_dropped.get_shape().as_list()[1], 10], stddev = 0.05, mean = 0.0))

        # one bias for each layer therefore 10 biasses
        output_layer_bias_tf = tf.Variable(tf.zeros(10))

        # logits also known as the output of the model
        logits = tf.add(tf.matmul(fully_connected_layer_2_dropped, output_layer_weights_tf), output_layer_bias_tf)

        return (x, y, keep_prob, logits)


    def train(self):
        epochs = 15

        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()

        x, y, keep_prob, logits = self.create_model()

        # loss of the neural network
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

        # optimizer will take in loss and minimize it using gradient descent
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Accuracy of the model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # TRAINING WILL BEGIN NOW
        print("Initiating tensorflow session...")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(epochs):
                training_batch = zip(range(0, len(self.train_x), 128), range(self.batch_size, len(self.train_x)+1, 128))

                for start, end in tqdm(training_batch):
                    sess.run(optimizer, feed_dict={x: self.train_x[start:end], y: self.train_y[start:end], keep_prob: self.keep_probability})


                # For printing purposes
                loss = sess.run(loss, feed_dict={x: self.train_x[start:end], y: self.train_y[start:end], keep_prob: 1.})

                valid_acc = sess.run(accuracy, feed_dict={x: self.test_x[:256], y: self.test_y[:256], keep_prob: 1.})

                print('Loss: {:>10.4f} Accuracy: {}'.format(loss, valid_acc))
