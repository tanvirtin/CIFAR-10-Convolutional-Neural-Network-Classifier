from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from one_hot_encode import *

class Preprocess(object):

    def load_data(self):
        print("Loading data...")

        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar10.load_data()

        self.num_train = len(self.train_x)
        self.num_test = len(self.test_x)

        # finding the length of the first row
        self.row = len(self.train_x[0])

        # finding the length of the first column of the first row
        self.col = len(self.train_x[0][0])

        # finding the length of the first element of the first row of the first column
        self.channel = len(self.train_x[0][0][0])

    def preprocess_images(self):
        print("Preprocessing images...")

        self.train_x = self.train_x.astype("float32") / 255
        self.test_x = self.test_x.astype("float32") / 255

    def one_hot_encode_labels(self):
        print("One hot encoding labels...")

        self.train_y = one_hot_encode(self.train_y)
        self.test_y = one_hot_encode(self.test_y)

        self.classes = len(self.train_y[0])

    # returns the number of training data, number of testing data, row, col and channel
    def metadata(self):
        print("Extracting metadata...")

        return (self.num_train, self.num_test, self.row, self.col, self.channel, self.classes)

    def get_training_data(self):
        print("Extracting training data...")

        return (self.train_x, self.train_y)

    def get_testing_data(self):
        print("Extracting testing data...")

        return (self.test_x, self.test_y)
