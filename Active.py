import copy
import numpy as np


class Active:
    def __init__(self, data, model, budget, target, balance):
        self.data = data
        self.model = model
        self.budget = budget
        self.target = target
        self.balance = balance
        self.f1_score = 0.0
        self.questions_asked = 0
        self.slide_uncertainty = []
        for i in range(len(self.data.data_y) - 1):
            temp = []
            for j in range(len(self.data.data_y[i]) - 1):
                temp.append(0)
            self.slide_uncertainty.append(temp)

    def new_model(self, verbose):
        model = copy.copy(self.model)
        model.verbose = verbose
        return model

    def train(self, data, verbose=True):
        if verbose:
            print('\n\nQuestions asked: ' + str(self.questions_asked))
            print('Data length: ' + str(len(data.train_y)))
        print('Data Balance: ' + str(data.check_balance()) + '\n')
        model = self.new_model(verbose)
        if self.model.loss_weights is not None:
            model.set_loss_params(weights=data.get_weights())
        return model.train(data)

    def train_predict(self, data, verbose=True):
        if verbose:
            print('\nQuestions asked: ' + str(self.questions_asked))
            print('Data length: ' + str(len(data.train_y)))
            print('Data Balance: ' + str(data.check_balance()))
        model = self.new_model(verbose)
        if self.model.loss_weights is not None:
            model.set_loss_params(weights=data.get_weights())
        accuracy, f1_score = model.train(data)
        return model.predict(data)

    def ranking(self, predictions, num_bootstraps):
        for i in range(len(predictions) - 1):
            for j in range(len(predictions[i]) - 1):
                index = np.argmax(predictions[i][j])
                temp_value = predictions[i][j][index]
                self.slide_uncertainty[i][j] += np.divide(temp_value, num_bootstraps)

    def get_index(self, batch_size):
        slide_average = np.zeros(len(self.slide_uncertainty))
        for i in range(len(self.slide_uncertainty) - 1):
            temp_array = []
            for j in range(len(self.slide_uncertainty[i]) - 1):
                temp_array.append(np.multiply(self.slide_uncertainty[i][j],
                                              (1 - self.slide_uncertainty[i][j])))
            slide_average[i] = sum(temp_array) / float(len(temp_array))
        return_list = []
        for _ in range(batch_size):
            return_list.append(np.argmax(slide_average))
            slide_average[np.argmax(slide_average)] = 0.0
        return return_list

    def run(self, num_bootstraps, bootstap_size, batch_size):
        f1_scores = []
        _, self.f1_score = self.train(self.data)
        f1_scores.append(self.f1_score)

        while self.budget != self.questions_asked and self.target > self.f1_score:
            self.slide_uncertainty = []
            for i in range(len(self.data.data_y) - 1):
                temp = []
                for j in range(len(self.data.data_y[i]) - 1):
                    temp.append(0)
                self.slide_uncertainty.append(temp)
            bootstraps = self.data.get_bootstraps(num_bootstraps, bootstap_size, 0.2, self.balance)
            for i in range(num_bootstraps):
                print('\nBootstrap: ' + str(i))
                predictions = self.train_predict(bootstraps[i], verbose=False)
                self.ranking(predictions, num_bootstraps)
            indices = self.get_index(batch_size)
            self.data.set_training_data(indices)
            self.questions_asked += 1
            _, self.f1_score = self.train(self.data)
            f1_scores.append(self.f1_score)

        return f1_scores
