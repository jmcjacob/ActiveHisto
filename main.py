import os
import cv2
import sys
import scipy.io as io
from Data import Data
from Model import Model
from Active import Active
from scipy.spatial import distance


def build_dataset(in_dir, out_dir, patch_size, steps, test_percentage):
    if in_dir[-1] != '/':
        in_dir += '/'
    if out_dir[-1] != '/':
        out_dir += '/'

    if not os.path.exists(out_dir + 'train'):
        os.makedirs(out_dir + 'train/miss')
        os.makedirs(out_dir + 'train/hit')
    if not os.path.exists(out_dir + 'test'):
        os.makedirs(out_dir + 'test/miss')
        os.makedirs(out_dir + 'test/hit')

    border_size = patch_size // 2
    train_count, test_count = 0, 0

    for img_no in range(1, len(os.listdir(in_dir)) + 1):
        original_image = cv2.imread(in_dir + 'img' + str(img_no) + '/img' + str(img_no) + '.bmp')
        border_image = cv2.copyMakeBorder(original_image, border_size, border_size, border_size, border_size,
                                          cv2.BORDER_DEFAULT)
        points = io.loadmat(in_dir + 'img' + str(img_no) + '/img' + str(img_no) + '_detection.mat')['detection']
        shape = original_image.shape
        patches_count, test_patches = 0, 10 - (10 * test_percentage)
        for i in range(0, shape[0], steps):
            for j in range(0, shape[1], steps):
                min_dist = 100
                for point in points:
                    dist = distance.euclidean((point[0], point[1]), (i, j))
                    if dist < min_dist:
                        min_dist = dist
                if min_dist <= 3:
                    cv2.rectangle(original_image, (i - border_size, j - border_size),
                                  (i + patch_size - border_size, j + patch_size - border_size), (255, 0, 0))
                    label = 'hit/'
                    cv2.imwrite(out_dir + '/' + str(img_no) + '.bmp', original_image)
                elif min_dist > 20:
                    label = 'miss/'
                else:
                    continue
                if patches_count > test_patches:
                    thing = 'test/'
                    test_count += 1
                else:
                    thing = 'train/'
                    train_count += 1
                patch = border_image[j:j + patch_size, i: i + patch_size]
                cv2.imwrite(out_dir + thing + label + str(img_no) + '_' + str(i) + ',' + str(j) + '.bmp', patch)

                if patches_count == 10:
                    patches_count = 0
                else:
                    patches_count += 1

                if (train_count + test_count) % 10000 == 0:
                    print(str(train_count + test_count) + ' Completed!')

        print('Image ' + str(img_no) + ' Completed!\n')
        print(str(train_count + test_count) + ' Completed!')
        cv2.imwrite(out_dir + '/' + str(img_no) + '.bmp', original_image)
    print('\nDone!')
    print('Training Data: ' + str(train_count))
    print('Testing Data: ' + str(test_count))


def train():
    data = Data(0.05)
    data.load_data(sys.argv[2])
    data.reduce_data(0.99)

    model = Model(1200, 2)
    model.set_loss_params(weights=data.get_weights())
    active = Active(data, model, 2, 1.00)
    f1_scores = active.run(5, 500, 100)
    print(f1_scores)

    # accuracy, f1 = model.train(data, intervals=1, epochs=1)
    # print('Accuracy: ' + str(accuracy))
    # print('F1-Score: ' + str(f1))


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build_dataset(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
    elif sys.argv[1] == 'train':
        train()
    else:
        print('build or train')
