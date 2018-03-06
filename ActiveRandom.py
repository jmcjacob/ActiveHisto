import copy
import random
from collections import Counter


class ActiveRandom:
    def __init__(self, data, model, budget):
        self.data = data
        self.model = model
        self.budget = budget
        self.questions = 0

    def new_model(self):
        return copy.copy(self.model)

    def train(self, data, epochs):
        model = self.new_model()
        return model.train(data, 'random', epochs=epochs)

    def update(self, update_size):
        if len(self.data.data_y) < 10:
            update_size = len(self.data.data_y)
        indices = list(range(len(self.data.data_y)))
        selection = random.sample(indices, update_size)
        self.data.set_training_data(selection, True)

    def run(self, update_size):
        f1_scores, accuracies, roc_areas, losses = [], [], [], []
        epochs = 500
        while len(self.data.data_y) != 0:
            print(Counter(self.data.train_y))
            accuracy, f1_score, roc_area, loss = self.train(self.data, epochs)
            f1_scores.append(f1_score)
            accuracies.append(accuracy)
            roc_areas.append(roc_area)
            losses.append(loss)
            self.update(update_size)
            self.questions += 1
            first = 20
        print(Counter(self.data.train_y))
        accuracy, f1_score, roc_area, loss = self.train(self.data, 100)
        f1_scores.append(f1_score)
        accuracies.append(accuracy)
        roc_areas.append(roc_area)
        losses.append(loss)
        print('F1 Scores: ' + str(f1_scores) + '\n')
        print('Accuracies: ' + str(accuracies) + '\n')
        print('ROCs: ' + str(roc_areas) + '\n')
        print('Losses: ' + str(losses))
        return f1_scores, accuracies, roc_areas, losses
