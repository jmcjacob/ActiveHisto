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
        for i in range(len(self.data.data_y)):
            temp = []
            for j in range(len(self.data.data_y[i])):
                temp.append(0)
            self.slide_uncertainty.append(temp)

    def new_model(self, verbose, bootstrap):
        model = copy.copy(self.model)
        model.verbose = verbose
        model.bootstrap = bootstrap
        return model

    def train(self, data, verbose=True):
        if verbose:
            print('\n\nQuestions asked: ' + str(self.questions_asked))
            print('Data length: ' + str(len(data.train_y)))
            print('Data Balance: ' + str(data.check_balance()) + '\n')
            print('\n\nQuestions asked: ' + str(self.questions_asked), file=open('results.txt', 'a'))
            print('Data length: ' + str(len(data.train_y)), file=open('results.txt', 'a'))
            print('Data Balance: ' + str(data.check_balance()) + '\n', file=open('results.txt', 'a'))
        model = self.new_model(verbose, False)
        if self.model.loss_weights is not None:
            model.set_loss_params(weights=data.get_weights())
        return model.train(data)

    def train_predict(self, data, verbose=True):
        if verbose:
            print('\n\nQuestions asked: ' + str(self.questions_asked))
            print('Data length: ' + str(len(data.train_y)))
            print('Data Balance: ' + str(data.check_balance()) + '\n')
            print('\n\nQuestions asked: ' + str(self.questions_asked), file=open('results.txt', 'a'))
            print('Data length: ' + str(len(data.train_y)), file=open('results.txt', 'a'))
            print('Data Balance: ' + str(data.check_balance()) + '\n', file=open('results.txt', 'a'))
        model = self.new_model(verbose, True)
        if self.model.loss_weights is not None:
            model.set_loss_params(weights=data.get_weights())
        accuracy, f1_score = model.train(data)
        return model.predict(data)

    def ranking(self, predictions, num_bootstraps):
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                index = np.argmax(predictions[i][j])
                temp_value = predictions[i][j][index]
                self.slide_uncertainty[i][j] += np.divide(temp_value, num_bootstraps)

    def get_index(self, batch_size):
        slide_max = np.zeros(len(self.slide_uncertainty))
        for i in range(len(self.slide_uncertainty)):
            temp_array = []
            for j in range(len(self.slide_uncertainty[i])):
                temp_array.append(np.multiply(self.slide_uncertainty[i][j],
                                              (1 - self.slide_uncertainty[i][j])))
            if len(temp_array) != 0:
                slide_max[i] = max(temp_array)
            else:
                print(i)
                print(len(self.slide_uncertainty))
                print(j)
                print(len(self.slide_uncertainty[i]))
                print('Something is wrong!')
        return_list = []
        for _ in range(batch_size):
            return_list.append(np.argmax(slide_max))
            slide_max[np.argmax(slide_max)] = 0.0
        return return_list

    def run(self, num_bootstraps, bootstap_size, batch_size):
        for i in range(len(self.data.data_x)):
            if len(self.data.data_x[i]) == 0:
                print(i)
        f1_scores = []
        _, self.f1_score = self.train(self.data)
        f1_scores.append(self.f1_score)

        while self.budget != self.questions_asked and self.target > self.f1_score:
            self.slide_uncertainty = []
            for i in range(len(self.data.data_y)):
                temp = []
                for j in range(len(self.data.data_y[i])):
                    temp.append(0)
                self.slide_uncertainty.append(temp)
            bootstraps = self.data.get_bootstraps(num_bootstraps, bootstap_size, 0.2, self.balance)
            for i in range(num_bootstraps):
                print('\nBootstrap: ' + str(i))
                print('\nBootstrap: ' + str(i), file=open('results.txt', 'a'))
                predictions = self.train_predict(bootstraps[i], verbose=False)
                self.ranking(predictions, num_bootstraps)
            indices = self.get_index(batch_size)
            self.data.set_training_data(indices)
            self.questions_asked += 1
            _, self.f1_score = self.train(self.data)
            f1_scores.append(self.f1_score)

        return f1_scores
