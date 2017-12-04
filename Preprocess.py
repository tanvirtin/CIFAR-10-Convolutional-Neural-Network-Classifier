from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

class Preprocess(object):
    def __init__(self):
        # the training and testing features and their labels
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def load_data(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar10.load_data()

    def plot_images(self):
        class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

        fig = plt.figure(figsize = (8, 3))

        for i in range(len(class_names)):
            ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
            idx = np.where(train_labels[:]==i)[0]
            features_idx = train_features[idx,::]
            img_num = np.random.randint(features_idx.shape[0])
            im = np.transpose(features_idx[img_num,::], (1, 2, 0))
            ax.set_title(class_names[i])
            plt.imshow(im)

        plt.show()
