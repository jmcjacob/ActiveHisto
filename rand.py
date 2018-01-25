import copy
import random


class Rand:
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

    def train(self, data, load):
        model = self.new_model(self.model.verbose, self.model.bootstrap)
        # if self.model.loss_weights is not None:
        #     model.set_loss_params(weights=weights)
        # self.model.redefine()
        accuracy, f1_score, roc = self.model.train(data, load=load)
        return accuracy, f1_score, roc

    def update(self, update_size):
        indices = list(range(len(self.data.data_y)))
        selection = random.sample(indices, update_size)
        self.data.set_training_data(selection)

    def run(self, update_size):
        f1_scores, accuracies, rocs = [], [], []
        load = False
        while self.budget != self.questions:
            acc, f1_score, roc = self.train(self.data, load)
            f1_scores.append(f1_score)
            accuracies.append(acc)
            rocs.append(roc)
            self.update(update_size)
            self.questions += 1
            # load = True
        print('F1 Scores: ' + str(f1_scores))
        print('Accuracies: ' + str(accuracies))
        print('ROCs: ' + str(rocs))
        return f1_scores, accuracies
