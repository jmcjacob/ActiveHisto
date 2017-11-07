import copy
import numpy as np


class Active:
    def __init__(self, data, model, budget, target):
        self.data = data
        self.model = model
        self.budget = budget
        self.target = target
        self.f1_score = 0.0
        self.questions_asked = 0
        self.slide_uncertainty = np.zeros((len(self.data.data_y)))

    def new_model(self):
        return copy.copy(self.model)

    def train(self, data, verbose=True):
        if verbose:
            print('\nQuestions asked: ' + str(self.questions_asked))
            print('\nData length: ' + str(len(data.train_y)))
        model = self.new_model()
        if not self.model.loss_weights.shape == np.ones((0, 0)).shape:
            model.set_loss_params(weights=data.get_weights())
        return model.train(data)

    def train_predict(self, data, verbose=True):
        if verbose:
            print('\nQuestions asked: ' + str(self.questions_asked))
            print('\nData length: ' + str(len(data.train_y)))
        model = self.new_model()
        if not self.model.loss_weights.shape == np.ones((0, 0)).shape:
            model.set_loss_params(weights=data.get_weights())
        accuracy, f1_score = model.train(data)
        return accuracy, f1_score, model.predict(data)
