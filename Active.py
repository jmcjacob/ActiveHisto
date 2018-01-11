import copy
import rand
import numpy as np
from collections import Counter


class Active:
    def __init__(self, data, model, budget):
        self.data = data
        self.model = model
        self.budget = budget
        self.questions = 0

    def new_model(self, verbose, bootstrap):
        model = copy.copy(self.model)
        model.verbose = verbose
        model.bootstrap = bootstrap
        return model

    def train(self, data, verbose, bootstrap, weights=None):
        model = self.new_model(verbose, bootstrap)
        if self.model.loss_weights is not None:
            model.set_loss_params(weights=weights)
        accuracy, f1_score = model.train(data)
        return accuracy, f1_score

    def train_predict(self, data, verbose, bootstrap, weights=None):
        model = self.new_model(verbose, bootstrap)
        if self.model.loss_weights is not None:
            model.set_loss_params(weights=weights)
        accuracy, f1_score = model.train(data)
        print('Number of predictions: ' + str(len(data.data_y)))
        predictions = model.predict(data)
        return predictions, accuracy, f1_score

    def shortlist(self, predictions, length):
        regions = []
        for region in predictions:
            temp = []
            for patch in region:
                temp.append(1 - max(patch))
            regions.append(max(temp))
        if True:
            shortlist = [i[1] for i in sorted(((value, index) for index, value in enumerate(regions)),
                                          reverse=True)[:length]]
        else:
            shortlist = []
            while len(shortlist) < length:
                for i in range(len(regions)):
                    value = rand.rand()
                    if regions[i] > value:
                        shortlist.append(i)
            while len(shortlist) > length:
                del shortlist[rand.randint(0, len(shortlist) - 1)]
        return shortlist

    def selection(self, bootstrap_predictions, shortlist, update_size):
        avg_predictions = []
        for i in range(len(bootstrap_predictions[0])):
            temp = 0
            for j in range(len(bootstrap_predictions)):
                for prediction in bootstrap_predictions[j][i]:
                    temp += (1 - max(prediction))
            avg_predictions.append(temp / len(bootstrap_predictions[0]))
        indices = np.array(avg_predictions).argsort()[-update_size:][::-1]
        return_list = []
        for index in indices:
            return_list.append(shortlist[index])
        return return_list

    def run(self, num_bootstraps, bootstrap_size, update_size):
        f1_scores = []
        while self.budget != self.questions:
            predictions, _, f1_score = self.train_predict(self.data, True, False, self.data.get_weights())
            f1_scores.append(f1_score)
            shortlist = self.shortlist(predictions, 500)
            bootstraps = self.data.get_bootstraps(num_bootstraps, bootstrap_size, 0.2, False)
            bootstrap_predictions = []
            for i in range(num_bootstraps):
                print('\nBootstrap: ' + str(i))
                bootstraps[i].reduce_data(shortlist)
                print('Bootstrap Data Balance ' + str(Counter(bootstraps[i].train_y)))
                predictions, _, _ = self.train_predict(bootstraps[i], False, True, bootstraps[i].get_weights())
                bootstrap_predictions.append(predictions)
            print('\n')
            indices = self.selection(bootstrap_predictions, shortlist, update_size)
            self.data.set_training_data(indices)
            self.questions += 1
        print(f1_scores)
