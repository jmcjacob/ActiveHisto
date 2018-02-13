import sys
from Model import Model
from Dataset import Data
import matplotlib.pyplot as plt
from ActiveRandom import ActiveRandom


def random(data, model):
    active_random = ActiveRandom(data, model, 10)
    return active_random.run(10)


def plotting(title, values, start, incriment):
    plt.plot(list(range(start, incriment * len(values) + start, incriment)), values)
    plt.xlabel('Regions')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(title + '.png')
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    if sys.argv[1] == 'supervised':
        f1_scores, accuracies, roc_areas, losses = [], [], [], []
        for i in range(10):
            data = Data(0.1)
            data.load_data(sys.argv[2])
            data.set_test_data(0.2)
            data.set_random_data(len(data.data_y), True)
            model = Model([15, 15, 3], 2)
            accuracy, f1_score, roc_area, loss = model.train(data, 'supervised')
            accuracies.append(accuracy)
            f1_scores.append(f1_score)
            roc_areas.append(roc_area)
            losses.append(loss)
        print('F1 Scores: ' + str(f1_scores) + '\n')
        print('Accuracies: ' + str(accuracies) + '\n')
        print('ROCs: ' + str(roc_areas) + '\n')
        print('Losses: ' + str(losses))
    else:
        data = Data(0.1)
        data.load_data(sys.argv[2])
        data.set_test_data(0.2)
        data.set_random_data(50, True)

        model = Model([15, 15, 3], 2)

        if sys.argv[1] == "random":
            f1_scores, accuracies, roc_areas, losses = random(data, model)

        plotting(sys.argv[1] + '/Accuracy', accuracies, 50, 10)
        plotting(sys.argv[1] + '/F1-Score', f1_scores, 50, 10)
        plotting(sys.argv[1] + '/ROC Area', roc_areas, 50, 10)
        plotting(sys.argv[1] + '/Loss', losses, 50, 10)
