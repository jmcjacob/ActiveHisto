import os
import random
import numpy as np
import tensorflow as tf
from collections import Counter
import tensorflow.contrib.data as tf_data
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, validation_percentage):
        self.train_x, self.train_y = [], []
        self.test_x, self.test_y = [], []
        self.val_x, self.val_y = [], []
        self.data_x, self.data_y = [], []
        self.validation_percentage = validation_percentage

    def load_data(self, data_dir):
        for image_dir in os.listdir(data_dir):
            if os.path.isdir(data_dir + image_dir):
                temp_x, temp_y = [], []
                for image_file in os.listdir(data_dir + image_dir + '/positive'):
                    temp_x.append(data_dir + image_dir + '/positive/' + image_file)
                    temp_y.append(1)
                for image_file in os.listdir(data_dir + image_dir + '/negative'):
                    temp_x.append(data_dir + image_dir + '/negative/' + image_file)
                    temp_y.append(0)
                self.data_x.append(temp_x)
                self.data_y.append(temp_y)

    def set_test_data(self, percentage):
        num_regions = int(len(self.data_y) * percentage)
        for _ in range(num_regions):
            index = random.randint(0, len(self.data_y) - 1)
            self.test_x += self.data_x[index]
            self.test_y += self.data_y[index]
            del self.data_x[index]
            del self.data_y[index]

    def set_random_data(self, num_regions, balanced=False):
        for _ in range(num_regions):
            index = random.randint(0, len(self.data_y) - 1)
            self.train_x += self.data_x[index]
            self.train_y += self.data_y[index]
            del self.data_x[index]
            del self.data_y[index]
        if balanced:
            balance = Counter(self.train_y)
            indices = []
            if balance[0] > balance[1]:
                number = balance[0] - balance[1]
                indices = np.random.permutation([i for i, j in enumerate(self.train_y) if j == 0])[:number]
            elif balance[0] < balance[1]:
                number = balance[1] - balance[0]
                indices = np.random.permutation([i for i, j in enumerate(self.train_y) if j == 1])[:number]
            self.train_y = [i for j, i in enumerate(self.train_y) if j not in indices]
            self.train_x = [i for j, i in enumerate(self.train_x) if j not in indices]
        # self.set_validation_set(balanced=balanced)

    def set_validation_set(self, balanced=False):
        self.train_x += self.val_x
        self.train_y += self.val_y
        self.val_x, self.val_y = [], []
        if not balanced:
            self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y,
                                                                                  test_size=self.validation_percentage)
        else:
            size = int(len(self.train_y) * self.validation_percentage)
            if size % 2 == 1:
                size -= 1
            size = size // 2
            indices = np.random.permutation([i for i, j in enumerate(self.train_y) if j == 0])[:size]
            indices += np.random.permutation([i for i, j in enumerate(self.train_y) if j == 1])[:size]
            indices = sorted(indices)

            self.val_x += [i for j, i in enumerate(self.train_x) if j in indices]
            self.val_y += [i for j, i in enumerate(self.train_y) if j in indices]
            self.train_y = [i for j, i in enumerate(self.train_y) if j not in indices]
            self.train_x = [i for j, i in enumerate(self.train_x) if j not in indices]

    def set_training_data(self, indices, balanced=False):
        temp_x, temp_y = [], []
        for index in sorted(indices, reverse=True):
            temp_x += self.data_x[index]
            temp_y += self.data_y[index]
            self.data_x.pop(index)
            self.data_y.pop(index)
        if balanced:
            balance = Counter(temp_y)
            while balance[0] != balance[1]:
                if balance[0] > balance[1]:
                    index = random.choice([i for i, j in enumerate(temp_y) if j == 0])
                elif balance[0] < balance[1]:
                    index = random.choice([i for i, j in enumerate(temp_y) if j == 1])
                del temp_y[index]
                del temp_x[index]
                balance = Counter(temp_y)
        self.train_x += temp_x
        self.train_y += temp_y
        # self.set_validation_set(balanced)

    def get_num_batches(self, train_batch_size, test_batch_size):
        num_train_batches = int(np.ceil(len(self.train_y) / train_batch_size))
        num_test_batches = int(np.ceil(len(self.test_y) / test_batch_size))
        num_val_batches = int(np.ceil(len(self.val_y) / test_batch_size))
        return num_train_batches, num_test_batches, num_val_batches

    def input_parser(self, image_path, label):
        one_hot_label = tf.one_hot(tf.to_int32(label), 2)
        image_file = tf.read_file(image_path)
        image = tf.image.decode_image(image_file, channels=3)
        return tf.cast(image, 'float'), one_hot_label

    def get_datasets(self, num_threads, buffer_size, train_batch_size, test_batch_size):
        train_images = tf.constant(np.asarray(self.train_x))
        train_labels = tf.constant(np.asarray(self.train_y))
        train_dataset = tf_data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.map(self.input_parser, num_threads, buffer_size)

        test_images = tf.constant(np.asarray(self.test_x))
        test_labels = tf.constant(np.asarray(self.test_y))
        test_dataset = tf_data.Dataset.from_tensor_slices((test_images, test_labels))
        test_dataset = test_dataset.map(self.input_parser, num_threads, buffer_size)

        # val_images = tf.constant(np.asarray(self.val_x))
        # val_labels = tf.constant(np.asarray(self.val_y))
        # val_dataset = tf_data.Dataset.from_tensor_slices((val_images, val_labels))
        # val_dataset = val_dataset.map(self.input_parser, num_threads, buffer_size)

        train_dataset = train_dataset.batch(train_batch_size)
        test_dataset = test_dataset.batch(test_batch_size)
        # val_dataset = val_dataset.batch(test_batch_size)

        return train_dataset.shuffle(buffer_size), test_dataset #, val_dataset.shuffle(buffer_size)

    def get_num_predict_batches(self, batch_size):
        count = 0
        for data in self.data_y:
            count += len(data)
        return int(np.ceil(count / batch_size))

    def predict_input_parser(self, image_path):
        image_file = tf.read_file(image_path)
        image = tf.image.decode_image(image_file, channels=3)
        return tf.cast(image, 'float')

    def get_predictions_dataset(self, num_threads, buffer_size, batch_size):
        indices, files = [], []
        for slide in self.data_x:
            files += slide
            indices.append(len(files))
        images = tf.constant(files)
        predict_dataset = tf_data.Dataset.from_tensor_slices((images))
        predict_dataset = predict_dataset.map(self.predict_input_parser, num_threads, buffer_size)
        return predict_dataset.batch(batch_size), indices
