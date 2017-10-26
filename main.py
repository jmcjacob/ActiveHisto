import os
import cv2
import math
import scipy.io as io
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

        # for point in mat:
        #     cv2.circle(orginal_image, (int(point[0]), int(point[1])), 1, (0, 0, 255))
        # cv2.imshow('image', orginal_image)
        # cv2.waitKey(0)

        patches, labels = [], []
        for i in range(1, shape[0], steps):
            for j in range(1, shape[1], steps):
                min_dist = 2147483647
                for point in mat:
                    dist = (math.sqrt(abs(point[0] - (i + border_size)) ** 2)) + \
                           (math.sqrt(abs(point[1] - (j + border_size)) ** 2))
                    if dist < min_dist:
                        min_dist = dist
                if 5 > min_dist and min_dist > 1:
                    continue
                if min_dist >= 5:
                    patches.append(image[i:i + patch_size, j:j + patch_size])
                    labels.append(0)
                else:
                    patches.append(image[i:i + patch_size, j:j + patch_size])
                    labels.append(1)
                if len(patches) == 100:
                    train_x, test_x, train_y, test_y = train_test_split(patches, labels, test_size=int(100 * test_percentage))
                    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=int(100 * validation_percentage))
                    for i in range(len(train_x)):
                        if train_y[i] == 0:
                            cv2.imwrite(out_dir + '/train/miss/' + str(train_counter) + '.bmp', train_x[i])
                        else:
                            cv2.imwrite(out_dir + '/train/hit/' + str(train_counter) + '.bmp', train_x[i])
                        train_counter += 1
                    for i in range(len(test_x)):
                        if test_y[i] == 0:
                            cv2.imwrite(out_dir + '/test/miss/' + str(train_counter) + '.bmp', test_x[i])
                        else:
                            cv2.imwrite(out_dir + '/test/hit/' + str(train_counter) + '.bmp', test_x[i])
                        test_counter += 1
                    for i in range(len(val_x)):
                        if val_y[i] == 0:
                            cv2.imwrite(out_dir + '/validation/miss/' + str(train_counter) + '.bmp', val_x[i])
                        else:
                            cv2.imwrite(out_dir + '/validation/hit/' + str(train_counter) + '.bmp', val_x[i])
                        val_counter += 1
                    patches, labels = [], []
                    if (train_counter + test_counter + val_counter) % 10000 == 0:
                        print(str(train_counter + test_counter + val_counter) + ' Completed!')
        print('\nImage ' + str(img_no) + ' Completed!')
        print(str(train_counter + test_counter + val_counter) + ' Completed!')
    print('Done!')
    print('Training Data: ' + str(train_counter))
    print('Testing Data: ' + str(test_counter))
    print('Validation Data: ' + str(val_counter))


def main():
    build_dataset('../Detection_data', '/media/jacob/1742-0054/data', 10, 2, 0.2, 0.1)


if __name__ == '__main__':
    main()
