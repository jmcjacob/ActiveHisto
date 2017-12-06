import copy


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

    def run(self, num_bootstraps, bootstrap_size, update_size):
        f1_scores = []
        predictions, _, f1_score = self.train_predict()
