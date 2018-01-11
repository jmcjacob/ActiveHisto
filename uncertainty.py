import copy


class Uncertainty:
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

    def train_predict(self, data, verbose, bootstrap, weights=None):
        model = self.new_model(verbose, bootstrap)
        if self.model.loss_weights is not None:
            model.set_loss_params(weights=weights)
        accuracy, f1_score = model.train(data)
        print('Number of predictions: ' + str(len(data.data_y)))
        predictions = model.predict(data)
        return predictions, accuracy, f1_score

    def update(self, predictions, update_size):
        regions = []
        for region in predictions:
            temp = []
            for patch in region:
                temp.append(1 - max(patch))
            regions.append(max(temp))
        selection = [i[1] for i in sorted(((value, index) for index, value in enumerate(regions)),
                                          reverse=True)[:update_size]]
        self.data.set_training_data(selection)

    def run(self, update_size):
        f1_scores = []
        accuracies = []
        while self.budget != self.questions:
            predictions, acc, f1_score = self.train_predict(self.data, True, False, self.data.get_weights())
            f1_scores.append(f1_score)
            accuracies.append(acc)
            self.update(predictions, update_size)
            self.questions += 1
        print('F1 Scores: ' + str(f1_scores))
        print('Accuracies: ' + str(accuracies))
        return f1_scores, accuracies
