import copy
import numpy as np


class Active:
    def __init__(self, data, model, budget, quality):
        self.data = data
        self.model = model
        self.budget = budget
        self.quality = quality
        self.f1 = 0.0
        self.questions_asked = 0
        self.values = np.zeros((len(self.data.predict_x)))

    def new_model(self):
        return copy.copy(self.model)

    def train_predict(self, data, verbose=True):
        if verbose:
            print('\nQuestions asked: ' + str(self.questions_asked))
            print('Data length: ' + str(len(data.train_x)))
        model = self.new_model()
        if not self.model.loss_weights.shape == np.ones((0, 0)).shape:
            model.set_loss_params(weights=data.get_weights())
        accuracy, f1 = model.train(data)
        return f1, model.predict(data)

    def ranking(self, predictions, number_bootstraps):
        for j in range(len(predictions)):
            index = np.argmax(predictions[j])
            temp = predictions[j][index]
            self.values[j] += np.divide(temp, number_bootstraps)
            self.values[j] = np.multiply(self.values[j], (1 - self.values[j]))

    def get_indexes(self, batch_size):
        indexes = []
        if self.data.balance:
            num_to_label = int(batch_size / 10)
            classification = [[], [], [], [], [], [], [], [], [], []]
            for i in range(len(self.values)):
                prediction_class = int(np.argmax(self.data.predict_y[i]))
                classification[prediction_class].append([self.values[i], i])
            for class_maxes in classification:
                c_maxes, big_indexes = [m[0] for m in class_maxes], [n[1] for n in class_maxes]
                for i in range(num_to_label):
                    index = np.where(c_maxes == np.asarray(c_maxes).min())[0][0]
                    class_maxes[index][0] = np.finfo(np.float64).max
                    index = big_indexes[index]
                    indexes.append(index)
        else:
            for i in range(batch_size):
                index = np.where(self.values == self.values.min())[0][0]
                self.values[index] = np.finfo(np.float64).max
                indexes.append(index)
        return indexes

    def run(self, number_bootstraps, bootstrap_size, batch_size):
        f1s = []
        self.f1, _ = self.train_predict(self.data)
        f1s.append(self.f1)

        while self.budget != self.questions_asked and self.quality > self.f1:
            self.values = np.zeros((len(self.data.predict_x)))
            bootstraps = self.data.get_bootstraps(number_bootstraps, bootstrap_size)
            for i in range(len(bootstraps)):
                print('\nBootstrap: ' + str(i))
                _, predictions = self.train_predict(bootstraps[i], verbose=False)
                self.ranking(predictions, number_bootstraps)
            indexes = self.get_indexes(batch_size)
            self.data.increase_data(indexes)
            self.questions_asked = self.questions_asked + 1
            self.f1, _ = self.train_predict(self.data)
            f1s.append(self.f1)

        return f1s