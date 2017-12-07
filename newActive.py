import copy
import random


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
        predictions = model.predict(data)
        return predictions, accuracy, f1_score

    def shortlist(self, predictions, length):
        for region in predictions:
            regions = []
            for patch in region:
                regions.append(min(patch))
        if False:
            shortlist = [i[1] for i in sorted(((value, index) for index, value in enumerate(regions)),
                                          reverse=True)[:length]]
        else:
            shortlist = []
            while len(shortlist) < length:
                for i in range(len(regions)):
                    value = random.random()
                    if regions[i] > value:
                        shortlist.append(i)
            while shortlist > length:
                del shortlist[random.randint(len(shortlist)-1)]
        return shortlist

    def run(self, num_bootstraps, bootstrap_size, update_size):
        f1_scores = []
        predictions, _, f1_score = self.train_predict(self.data, True, False, self.data.get_weights())
        f1_scores.append(f1_score)
        shortlist = self.shortlist(predictions, 500)

        while self.budget != self.questions:
            bootstraps = self.data.get_bootstraps(num_bootstraps, bootstrap_size, 0.2, False)
            for i in range(num_bootstraps):
                print('\nBootstrap: ' + str(i))
                bootstraps[i].reduce_data(shortlist)
                predictions, _, _ = self.train_predict(bootstraps[i], False, True, bootstraps[i].get_weights())
                
