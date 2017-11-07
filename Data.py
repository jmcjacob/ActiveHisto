import os
import random
import numpy as np
import tensorflow as tf
from collections import Counter
import tensorflow.contrib.data as tf_data
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, val_percentage):
        self.train_x, self.train_y = [], []
        self.test_x, self.test_y = [], []
        self.val_x, self.val_y = [], []
        self.data_x, self.data_y = [], []
        self.val_percentage = val_percentage

    def load_test_data(self, test_dir):
        for image_dir in os.listdir(test_dir):
            if os.path.isdir(test_dir + image_dir):
                for image_file in os.listdir(test_dir + image_dir + '/hit'):
                    self.test_x.append(test_dir + image_dir + '/hit/' + image_file)
                    self.test_y.append(1)
                for image_file in os.listdir(test_dir + image_dir + '/miss'):
                    self.test_x.append(test_dir + image_dir + '/miss/' + image_file)
                    self.test_y.append(0)

    def load_data(self, data_dir):
        for image_dir in os.listdir(data_dir):
            if os.path.isdir(data_dir + image_dir):
                temp_x, temp_y = [], []
                for image_file in os.listdir(data_dir + image_dir + '/hit'):
                    temp_x.append(data_dir + image_dir + '/hit/' + image_file)
                    temp_y.append(1)
                for image_file in os.listdir(data_dir + image_dir + '/miss'):
                    temp_x.append(data_dir + image_dir + '/miss/' + image_file)
                    temp_y.append(0)
                self.data_x.append(temp_x)
                self.data_y.append(temp_y)

    def make_val_set(self, percentage):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y,
                                                                              test_size=percentage)

    def set_random_training_data(self, number):
        for _ in range(number):
            index = random.randint(len(self.data_y) - 1)
            self.train_x += self.data_x[index]
            self.train_y += self.data_y[index]
            self.data_x.remove(index)
            self.data_y.remove(index)

    def set_training_data(self, indices):
        if type(indices) != int:
            for index in indices:
                self.train_x += self.data_x[index]
                self.train_y += self.data_y[index]
                self.data_x.remove(index)
                self.data_y.remove(index)
        else:
            self.train_x += self.data_x[indices]
            self.train_y += self.data_y[indices]
            self.data_x.remove(indices)
            self.data_y.remove(indices)

    def set_data(self, train_x, train_y):
        self.train_x, self.train_y = train_x, train_y
        self.make_val_set(self.val_percentage)

    def get_num_batches(self, train_batch_size, test_batch_size):
        train_batch = int(np.floor(len(self.train_y) / train_batch_size))
        test_batch = int(np.floor(len(self.test_y) / (test_batch_size)))
        val_batch = int(np.floor(len(self.val_y) / (test_batch_size)))
        return train_batch, test_batch, val_batch

    def get_num_predict_batches(self, batch_size, index):
        return int(np.floor(len(self.data_y[index]) / batch_size))

    def input_parser(self, image_path, label):
        one_hot_label = tf.constant(tf.to_int32(label), 2)
        image_file = tf.read_file(image_path)
        image = tf.image.decode_image(image_file, channels=3)
        return tf.reshape(image, [1200]), one_hot_label

    def predict_input_parser(self, image_path):
        image_file = tf.read_file(image_path)
        image = tf.image.decode_image(image_file, channels=3)
        return tf.reshape(image, [1200])

    def get_datasets(self, num_threads, buffer_size, train_batch_size, test_batch_size):
        train_images = tf.constant(self.train_x)
        train_labels = tf.constant(self.train_y)
        train_dataset = tf_data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.map(self.input_parser, num_threads, buffer_size)

        test_images = tf.constant(self.test_x)
        test_labels = tf.constant(self.test_y)
        test_dataset = tf_data.Dataset.from_tensor_slices((test_images, test_labels))
        test_dataset = test_dataset.map(self.input_parser, num_threads, buffer_size)

        validation_images = tf.constant(self.val_x)
        validation_labels = tf.constant(self.val_y)
        validation_dataset = tf_data.Dataset.from_tensor_slices((validation_images, validation_labels))
        validation_dataset = validation_dataset.map(self.input_parser, num_threads, buffer_size)

        train_dataset = train_dataset.batch(train_batch_size)
        test_dataset = test_dataset.batch(test_batch_size)
        validation_dataset = validation_dataset.batch(test_batch_size)

        return train_dataset.shuffle(10000), test_dataset, validation_dataset

    def get_predict_dataset(self, num_threads, buffer_size, batch_size, index):
        images = tf.constant(self.data_x[index])
        predict_dataset = tf_data.Dataset.from_tensor_slices((images))
        predict_dataset = predict_dataset.map(self.predict_input_parser, num_threads, buffer_size)
        return predict_dataset.batch(batch_size)

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

    def get_bootstraps(self, num_bootstraps, bootstrap_size):
        bootstraps = []
        for _ in range(num_bootstraps):
            bootstrap_x, bootstrap_y = [], []
            indices = random.sample(range(len(self.train_x)), bootstrap_size)
            for index in indices:
                bootstrap_x.append(self.train_x[index])
                bootstrap_y.append(self.train_y[index])
            data = Data(self.val_percentage)
            data.set_data(bootstrap_x, bootstrap_y)
            bootstraps.append(data)
        return bootstraps
