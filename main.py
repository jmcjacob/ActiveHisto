import os
import sys
import Dataset
from Data import Data
from Model import Model
from Active import Active
from uncertainty import Uncertainty
from rand import Rand

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def supervised():
    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.load_test_data(sys.argv[3])

    # Train a single model with all of the Data.
    model = Model([27, 27, 3], 2)
    model.set_loss_params(weights=data.get_weights())
    accuracy, f1_score = model.train(data)
    print('Accuracy: ' + str(accuracy))
    print('Accuracy: ' + str(accuracy), file=open('results.txt', 'a'))
    print('F1-Score: ' + str(f1_score))
    print('F1-Score: ' + str(f1_score), file=open('results.txt', 'a'))


def bootstrap():
    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.load_test_data(sys.argv[3])

    # Train with Bootstraped Uncertainty.
    data.set_random_balanced_data(int(sys.argv[4]))
    model = Model([27, 27, 3], 2)
    model.set_loss_params(weights=data.get_weights())
    active = Active(data, model, 10)
    f1_scores, accurices = active.run(20, 100, int(sys.argv[4]))
    print('Update size: ' + sys.argv[4], file=open('results.txt', 'a'))
    print('F1 Scores: ' + str(f1_scores), file=open('results.txt', 'a'))
    print('Accuracies: ' + str(accurices), file=open('results.txt', 'a'))


def rand():
    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.load_test_data(sys.argv[3])

    #Train with random selection
    data.set_random_balanced_data(int(sys.argv[4]))
    model = Model([27, 27, 3], 2)
    model.set_loss_params(weights=data.get_weights())
    rand = Rand(data, model, 10)
    f1_scores, accurices = rand.run(int(sys.argv[4]))
    print('Update size: ' + sys.argv[4], file=open('results.txt', 'a'))
    print('F1 Scores: ' + str(f1_scores), file=open('results.txt', 'a'))
    print('Accuracies: ' + str(accurices), file=open('results.txt', 'a'))


def uncertainty():
    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.load_test_data(sys.argv[3])

    #Train with uncertanty selection
    data.set_random_balanced_data(int(sys.argv[4]))
    model = Model([27, 27, 3], 2)
    model.set_loss_params(weights=data.get_weights())
    uncertain = Uncertainty(data, model, 10)
    f1_scores, accurices = uncertain.run(int(sys.argv[4]))
    print('Update size: ' + sys.argv[4], file=open('results.txt', 'a'))
    print('F1 Scores: ' + str(f1_scores), file=open('results.txt', 'a'))
    print('Accuracies: ' + str(accurices), file=open('results.txt', 'a'))


if __name__ == '__main__':
    if sys.argv[1] == 'binary':
        Dataset.build_binary_detection_dataset(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
                                               int(sys.argv[6]))
    elif sys.argv[1] == 'supervised':
        supervised()
    elif sys.argv[1] == 'bootstrap':
        bootstrap()
    elif sys.argv[1] == 'random':
        rand()
    elif sys.argv[1] == 'uncertainty':
        uncertainty()
    else:
        print('classification, regression or train')
    print('\n', file=open('results.txt', 'a'))
