import os
import sys
import Dataset
from copy import copy
from Data import Data
from Model import Model
from Active import Active
from rand import Rand


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def bootstrap():
    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.load_test_data(sys.argv[3])

    # Train a single model with all of the Data.
    # tempdata = copy(data)
    # tempdata.set_training_data(range(len(tempdata.data_y)))
    # model = Model([26, 26, 3], 2)
    # model.set_loss_params(weights=tempdata.get_weights())
    # accuracy, f1_score = model.train(tempdata, epochs=1, batch_size=100, intervals=1)
    # print('Accuracy: ' + str(accuracy))
    # print('Accuracy: ' + str(accuracy), file=open('results.txt', 'a'))
    # print('F1-Score: ' + str(f1_score))
    # print('F1-Score: ' + str(f1_score), file=open('results.txt', 'a'))

    # Train with Bootstraped Uncertainty.
    data.set_random_balanced_data(5)
    model = Model([27, 27, 3], 2)
    model.set_loss_params(weights=data.get_weights())
    active = Active(data, model, 10)
    f1_scores = active.run(20, 100, 5)
    print(f1_scores)


def rand():
    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.load_test_data(sys.argv[3])

    #Train with random selection
    data.set_random_balanced_data(5)
    model = Model([27, 27, 3], 2)
    model.set_loss_params(weights=data.get_weights())
    rand = Rand(data, model, 10)
    rand.run(5)


if __name__ == '__main__':
    if sys.argv[1] == 'binary':
        Dataset.build_binary_detection_dataset(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
                                               int(sys.argv[6]))
    elif sys.argv[1] == 'bootstrap':
        bootstrap()
    elif sys.argv[1] == 'random':
        rand()
    else:
        print('classification, regression or train')
