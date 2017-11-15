import os
import sys
import Dataset
from copy import copy
from Data import Data
from Model import Model
from Active import Active


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train():
    if sys.argv[4].lower() == 'true':
        balance = True
    else:
        balance = False

    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.load_test_data(sys.argv[3])

    # Train a single model with all of the Data.
    tempdata = copy(data)
    tempdata.set_training_data(range(len(tempdata.data_y)))
    model = Model([29, 29], 2)
    model.set_loss_params(weights=tempdata.get_weights())
    accuracy, f1_score = model.train(tempdata, epochs=-1, intervals=10, batch_size=100)
    print('Accuracy: ' + str(accuracy))
    print('Accuracy: ' + str(accuracy), file=open('results.txt', 'a'))
    print('F1-Score: ' + str(f1_score))
    print('F1-Score: ' + str(f1_score), file=open('results.txt', 'a'))

    # Train with Active Learning.
    data.set_random_training_data(1)
    model = Model([2523], 2)
    model.set_loss_params(weights=data.get_weights())
    active = Active(data, model, 10, 1., balance)
    f1_scores = active.run(100, 5000, 1)
    print(f1_scores)


if __name__ == '__main__':
    if sys.argv[1] == 'binary':
        Dataset.build_binary_detection_dataset(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[1] == 'train':
        train()
    else:
        print('classification, regression or train')
