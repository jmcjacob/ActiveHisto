import os
import cv2
import csv
import sys
import scipy.io as io
from Data import Data
from scipy.spatial import distance
from sklearn.model_selection import train_test_split


def build_dataset(in_dir, out_dir, patch_size, steps, test_percentage, validation_percentage):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir + '/train'):
        os.makedirs(out_dir + '/train/miss')
        os.makedirs(out_dir + '/train/hit')
    if not os.path.exists(out_dir + '/test'):
        os.makedirs(out_dir + '/test/miss')
        os.makedirs(out_dir + '/test/hit')
    if not os.path.exists(out_dir + '/validation'):
        os.makedirs(out_dir + '/validation/miss')
        os.makedirs(out_dir + '/validation/hit')

    border_size = patch_size // 2
    train_counter, test_counter, val_counter = 0, 0, 0

    for img_no in range(1, 101):
        orginal_image = cv2.imread(in_dir + '/img' + str(img_no) + '/img' + str(img_no) + '.bmp')
        shape = orginal_image.shape
        image = cv2.copyMakeBorder(orginal_image, border_size, border_size, border_size, border_size,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        mat = io.loadmat(in_dir + '/img' + str(img_no) + '/img' + str(img_no) + '_detection.mat')['detection']

        patches, labels = [], []
        for i in range(1, shape[0], steps):
            for j in range(1, shape[1], steps):
                min_dist = 2147483647
                for point in mat:
                    dist = distance.euclidean(point, (i, j))
                    if abs(dist) < min_dist:
                        min_dist = abs(dist)
                if 20 > min_dist and min_dist > 3:
                    continue
                if min_dist >= 20:
                    patches.append(image[i:i + patch_size, j:j + patch_size])
                    labels.append(0)
                else:
                    patches.append(image[i:i + patch_size, j:j + patch_size])
                    labels.append(1)
                if len(patches) == 100:
                    train_x, test_x, train_y, test_y = train_test_split(patches, labels,
                                                                        test_size=int(100 * test_percentage))
                    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                                      test_size=int(100 * validation_percentage))
                    for i in range(len(train_x)):
                        with open(out_dir + '/train.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            if train_y[i] == 0:
                                cv2.imwrite(out_dir + '/train/miss/' + str(train_counter) + '.bmp', train_x[i])
                                writer.writerow([out_dir + '/train/miss/' + str(train_counter) + '.bmp'] + [train_y[i]])
                            else:
                                cv2.imwrite(out_dir + '/train/hit/' + str(train_counter) + '.bmp', train_x[i])
                                writer.writerow([out_dir + '/train/hit/' + str(train_counter) + '.bmp'] + [train_y[i]])
                        train_counter += 1
                    for i in range(len(test_x)):
                        with open(out_dir + '/test.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            if test_y[i] == 0:
                                cv2.imwrite(out_dir + '/test/miss/' + str(test_counter) + '.bmp', test_x[i])
                                writer.writerow([out_dir + '/test/miss/' + str(test_counter) + '.bmp'] + [test_y[i]])
                            else:
                                cv2.imwrite(out_dir + '/test/hit/' + str(test_counter) + '.bmp', test_x[i])
                                writer.writerow([out_dir + '/test/hit/' + str(test_counter) + '.bmp'] + [test_y[i]])
                        test_counter += 1
                    for i in range(len(val_x)):
                        with open(out_dir + '/validation.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            if val_y[i] == 0:
                                cv2.imwrite(out_dir + '/validation/miss/' + str(val_counter) + '.bmp', val_x[i])
                                writer.writerow([out_dir + '/validation/miss/' + str(val_counter) + '.bmp'] + [val_y[i]])
                            else:
                                cv2.imwrite(out_dir + '/validation/hit/' + str(val_counter) + '.bmp', val_x[i])
                                writer.writerow([out_dir + '/validation/hit/' + str(val_counter) + '.bmp'] + [val_y[i]])

                        val_counter += 1
                    patches, labels = [], []
                    if (train_counter + test_counter + val_counter) % 10000 == 0:
                        print(str(train_counter + test_counter + val_counter) + ' Completed!')
        print('Image ' + str(img_no) + ' Completed!\n')
        print(str(train_counter + test_counter + val_counter) + ' Completed!')
    print('Done!')
    print('Training Data: ' + str(train_counter))
    print('Testing Data: ' + str(test_counter))
    print('Validation Data: ' + str(val_counter))


def train():
    data = Data()
    data.load_data(sys.argv[2])

    # Train a model with all the data for benchmarking purposes

    data.reduce_data(0.99)


if __name__ == '__main__':
    print(sys.argv)
    if sys.argv[1] == 'build':
        try:
            build_dataset(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
        except:
            print('main.py build input_dir output_dir patch_size skip test_size validation_size')
    if sys.argv[1] == 'train':
        try:
            train()
        except:
            print('main.py train input_dir')
    else:
        print('build or train')
