import os
import random
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.contrib.data import Dataset
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, val_percentage):
        self.train_x, self.train_y = [], []
        self.test_x, self.test_y = [], []
        self.val_percentage = val_percentage

    def set_data(self, train_x, train_y):
        self.train_x, self.train_y = train_x, train_y
        self.make_val_set(self.val_percentage)

    def load_data(self, dir):
        if dir[-1] != '/':
            dir = dir + '/'
        for image_file in os.listdir(dir + 'train/hit'):
            self.train_x.append(dir + 'train/hit/' + image_file)
            self.train_y.append(1)
        for image_file in os.listdir(dir + 'train/miss'):
            self.train_x.append(dir + 'train/miss/' + image_file)
            self.train_y.append(0)

        for image_file in os.listdir(dir + 'test/hit'):
            self.test_x.append(dir + 'test/hit/' + image_file)
            self.test_y.append(1)
        for image_file in os.listdir(dir + 'test/miss'):
            self.test_x.append(dir + 'test/miss/' + image_file)
            self.test_y.append(0)
        self.make_val_set(self.val_percentage)

    def make_val_set(self, percentage):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y,
                                                                              test_size=percentage)

    def get_sizes(self, batch_size):
        train_batch = int(np.floor(len(self.train_y) / batch_size))
        test_batch = int(np.floor(len(self.test_y) / (batch_size * 100)))
        val_batch = int(np.floor(len(self.val_y) / (batch_size * 100)))
        return train_batch, test_batch, val_batch

    def get_predict_size(self, batch_size):
        return int(np.floor(len(self.predict_y) / batch_size))

    def get_datasets(self, num_threads, buffer_size, batch_size):
        train_imgs = tf.constant(self.train_x)
        train_labels = tf.constant(self.train_y)
        train_dataset = Dataset.from_tensor_slices((train_imgs, train_labels))
        train_dataset = train_dataset.map(self.input_parser, num_threads, buffer_size)

        test_imgs = tf.constant(self.test_x)
        test_labels = tf.constant(self.test_y)
        test_dataset = Dataset.from_tensor_slices((test_imgs, test_labels))
        test_dataset = test_dataset.map(self.input_parser, num_threads, buffer_size)

        val_imgs = tf.constant(self.val_x)
        val_labels = tf.constant(self.val_y)
        val_dataset = Dataset.from_tensor_slices((val_imgs, val_labels))
        val_dataset = val_dataset.map(self.input_parser, num_threads, buffer_size)

        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size * 100)
        val_dataset = val_dataset.batch(batch_size * 100)

        return train_dataset.shuffle(10000), test_dataset, val_dataset

    def get_prediction_dataset(self, num_threads, buffer_size, batch_size):
        imgs = tf.constant(self.predict_x)
        predict_dataset = Dataset.from_tensor_slices((imgs))
        predict_dataset = predict_dataset.map(self.predict_parser, num_threads, buffer_size)
        return predict_dataset.batch(batch_size)

    def input_parser(self, img_path, label):
        one_hot = tf.one_hot(tf.to_int32(label), 2)
        img_file = tf.read_file(img_path)
        img = tf.image.decode_image(img_file, channels=3)
        return tf.reshape(img, [1200]), one_hot

    def predict_parser(self, img_path):
        img_file = tf.read_file(img_path)
        img = tf.image.decode_image(img_file, channels=3)
        return tf.reshape(img, [1200])

    def get_weights(self):
        counter = Counter(self.train_y)
        majority = max(counter.values())
        weights = {cls: float(majority / count) for cls, count in counter.items()}

        nb_cl = len(weights)
        final_weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            final_weights[0][class_idx] = class_weight
            final_weights[class_idx][0] = class_weight
        return final_weights

    def reduce_data(self, percentage):
        self.train_x, self.predict_x, self.train_y, self.predict_y = train_test_split(self.train_x,
                                                                                          self.train_y,
                                                                                          test_size=percentage)

    def get_bootstraps(self, num_bootstraps, bootstrap_size):
        bootstraps = []
        for _ in range(num_bootstraps):
            bootstrap_x, bootstrap_y = [], []
            indexs = random.sample(range(len(self.train_x)), bootstrap_size)
            for index in indexs:
                bootstrap_x.append(self.train_x[index])
                bootstrap_y.append(self.train_y[index])
            while bootstrap_y[1:] == bootstrap_y[:-1]:
                index = random.randint(len(bootstrap_y) -1)
                bootstrap_x.remove(index)
                bootstrap_y.remove(index)
                # TO-DO Add element to dataset.
            data = Data(self.val_percentage)
            data.set_data(np.asarray(bootstrap_x), np.asarray(bootstrap_y))
            bootstraps.append(data)
        return bootstraps

    def increase_data(self, indexes):
        self.train_x += self.val_x
        self.train_y += self.val_y
        self.val_x, self.val_y = [], []
        for index in indexes:
            self.train_x = np.vstack((self.train_x, [self.predict_x[index]]))
            self.train_y = np.vstack((self.train_y, [self.predict_y[index]]))
        self.predict_x = np.delete(self.predict_x, indexes, axis=0)
        self.predict_y = np.delete(self.predict_y, indexes, axis=0)
        self.make_val_set(self.val_percentage)

