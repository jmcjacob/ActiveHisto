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

    def __copy__(self):
        data = Data(self.val_percentage)
        data.data_x = self.data_x + self.train_x + self.val_x
        data.data_y = self.data_y + self.train_y + self.val_y
        data.test_x, data.test_y = self.test_x, self.test_y
        return data

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
                if temp_x == []:
                    print('')
                self.data_x.append(temp_x)
                self.data_y.append(temp_y)
        lowest = 10000
        for i in range(len(self.data_x)):
            if len(self.data_x[i]) < lowest:
                lowest = len(self.data_x[i])
        print(lowest)

    def reduce_data(self, indices):
        temp_x, temp_y = [], []
        for i in range(len(self.data_x)):
            if i in indices:
                temp_x.append(self.data_x[i])
                temp_y.append(self.data_y[i])
        self.data_x = temp_x
        self.data_y = temp_y

    def make_val_set(self):
        self.train_x += self.val_x
        self.train_y += self.val_y
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y,
                                                                              test_size=self.val_percentage)

    def set_random_training_data(self, number):
        for _ in range(number):
            index = random.randint(0, len(self.data_y) - 1)
            self.train_x += self.data_x[index]
            self.train_y += self.data_y[index]
            self.data_x.pop(index)
            self.data_y.pop(index)
        self.make_val_set()

    def set_random_balanced_data(self, number):
        for i in range(len(self.data_x)):
            if len(self.data_x[i]) == 0:
                print('Thing: ' + str(i))
        for _ in range(number):
            index = random.randint(0, len(self.data_y) - 1)
            while len(Counter(self.data_y[index])) != 2:
                index = random.randint(0, len(self.data_y) - 1)
            self.train_x += self.data_x[index]
            self.train_y += self.data_y[index]
            del self.data_x[index]
            del self.data_y[index]
            for i in range(len(self.data_x)):
                if len(self.data_x[i]) == 0:
                    print(i)
        while Counter(self.train_y)[0] != Counter(self.train_y)[1]:
            if Counter(self.train_y)[0] > Counter(self.train_y)[1]:
                index = random.choice([i for i, j in enumerate(self.train_y) if j == 0])
            elif Counter(self.train_y)[0] < Counter(self.train_y)[1]:
                index = random.choice([i for i, j in enumerate(self.train_y) if j == 1])
            del self.train_y[index]
            del self.train_x[index]
        print('Data Balance: ' + str(Counter(self.train_y)))
        self.make_val_set()


    def set_training_data(self, indices):
        temp_x = []
        temp_y = []
        if type(indices) != int:
            for index in sorted(indices, reverse=True):
                temp_x += self.data_x[index]
                temp_y += self.data_y[index]
                self.data_x.pop(index)
                self.data_y.pop(index)
        else:
            temp_x += self.data_x[indices]
            temp_y += self.data_y[indices]
            self.data_x.pop(indices)
            self.data_y.pop(indices)
        while Counter(temp_y)[0] != Counter(temp_y)[1]:
            if Counter(temp_y)[0] > Counter(temp_y)[1]:
                index = random.choice([i for i, j in enumerate(temp_y) if j == 0])
            elif Counter(temp_y)[0] < Counter(temp_y)[1]:
                index = random.choice([i for i, j in enumerate(temp_y) if j == 1])
            del temp_y[index]
            del temp_x[index]
        if True:
            self.train_x += temp_x
            self.train_y += temp_y
        elif False:
            self.train_x = temp_x
            self.train_y = temp_y
        else:
            pass
        print('Data Balance: ' + str(Counter(self.train_y)))
        self.make_val_set()

    def set_data(self, train_x, train_y):
        self.train_x, self.train_y = train_x, train_y
        self.make_val_set()

    def get_num_batches(self, train_batch_size, test_batch_size):
        train_batch = int(np.ceil(len(self.train_y) / train_batch_size))
        test_batch = int(np.ceil(len(self.test_y) / (test_batch_size)))
        val_batch = int(np.ceil(len(self.val_y) / (test_batch_size)))
        return train_batch, test_batch, val_batch

    def get_num_predict_batches(self, batch_size):
        sum = 0
        for data in self.data_y:
            sum += len(data)
        return int(np.ceil(sum / batch_size))

    def input_parser(self, image_path, label):
        one_hot_label = tf.one_hot(tf.to_int32(label), 2)
        image_file = tf.read_file(image_path)
        image = tf.image.decode_image(image_file, channels=3)
        return tf.cast(image, 'float'), one_hot_label

    def predict_input_parser(self, image_path):
        image_file = tf.read_file(image_path)
        image = tf.image.decode_image(image_file, channels=3)
        return tf.cast(image, 'float')

    def get_datasets(self, num_threads, buffer_size, train_batch_size, test_batch_size):
        train_images = tf.constant(np.asarray(self.train_x))
        train_labels = tf.constant(np.asarray(self.train_y))
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

        return train_dataset.shuffle(buffer_size), test_dataset, validation_dataset

    def get_predict_dataset(self, num_threads, buffer_size, batch_size):
        indices, files = [], []
        for slide in self.data_x:
            if slide == []:
                print('Something has gone wrong! part 1')
            files += slide
            indices.append(len(files))
        images = tf.constant(files)
        predict_dataset = tf_data.Dataset.from_tensor_slices((images))
        predict_dataset = predict_dataset.map(self.predict_input_parser, num_threads, buffer_size)
        return predict_dataset.batch(batch_size), indices

    def get_weights(self):
        counter = Counter(self.train_y)
        return counter[0] / counter[1]

    def get_bootstraps(self, num_bootstraps, bootstrap_size, val_percentage, balance):
        bootstraps = []
        for _ in range(num_bootstraps):
            bootstrap_x, bootstrap_y = [], []
            if balance:
                classes = [[], []]
                for i in range(len(self.train_y)):
                    classes[int(self.train_y[i])].append(i)
                for classification in classes:
                    indices = np.random.choice(classification, bootstrap_size // 2, replace=True)
                    for index in indices:
                        bootstrap_x.append(self.train_x[index])
                        bootstrap_y.append(self.train_y[index])
            else:
                indices = np.random.choice(range(len(self.train_x)), bootstrap_size, replace=True)
                for index in indices:
                    bootstrap_x.append(self.train_x[index])
                    bootstrap_y.append(self.train_y[index])
            data = Data(val_percentage)
            data.set_data(bootstrap_x, bootstrap_y)
            data.test_x, data.test_y = self.test_x, self.test_y
            data.data_x, data.data_y = self.data_x, self.data_y
            bootstraps.append(data)
        return bootstraps

    def check_balance(self):
        return Counter(self.train_y)
