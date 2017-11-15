import os
import cv2
import scipy.io as io
from scipy.spatial import distance


def build_binary_detection_dataset(in_dir, out_dir, patch_size, steps):
    if in_dir[-1] != '/': in_dir += '/'
    if out_dir[-1] != '/': out_dir += '/'

    border_size, count = patch_size // 2, 0
    start = int(sorted(os.listdir(in_dir))[0].replace('img', ''))

    for img_num in range(start, start + len(os.listdir(in_dir))):
        if not os.path.exists(out_dir + str(img_num) + '/'):
            os.makedirs(out_dir + str(img_num) + '/hit/')
            os.makedirs(out_dir + str(img_num) + '/miss/')
        original_image = cv2.imread(in_dir + 'img' + str(img_num) + '/img' + str(img_num) + '.bmp')
        border_image = cv2.copyMakeBorder(original_image, border_size, border_size, border_size, border_size,
                                          cv2.BORDER_DEFAULT)
        points = io.loadmat(in_dir + 'img' + str(img_num) + '/img' + str(img_num) + '_detection.mat')['detection']
        for i in range(0, original_image.shape[0], steps):
            for j in range(0, original_image.shape[1], steps):
                min_dist = 1000
                for point in points:
                    dist = distance.euclidean((point[0], point[1]), (i, j))
                    if dist < min_dist: min_dist = dist
                if min_dist <= 3:
                    cv2.rectangle(original_image, (i - border_size, j - border_size),
                                  (i + border_size, j + border_size), (255, 0, 0))
                    label = '/hit/'
                elif min_dist > 20:
                    label = '/miss/'
                else:
                    continue
                patch = border_image[j:j + patch_size, i:i + patch_size]
                patches = [patch, cv2.flip(patch, 0), cv2.flip(patch, 1), cv2.flip(patch, 2),
                           cv2.GaussianBlur(patch, (3, 3), 0.25)]

                rows, cols = patch.shape
                for i in [90, 180, 260]:
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), i, 1)
                    patches.append(cv2.warpAffine(patch, M, (cols, rows)))

                patches = [patch, cv2.flip(patch, 0), cv2.flip(patch, 1), cv2.flip(patch, 2),
                           cv2.GaussianBlur(patch, (3, 3), 0.25), cv2.medianBlur(patch, 3)]
                for p in range(len(patches)):
                    cv2.imwrite(out_dir + str(img_num) + label + str(i) + ',' + str(j) + str(p) + '.bmp',
                                patches[p])
                    count += 1

                if count % 10000 == 0:
                    print(str(count) + ' Completed!')
        print('Image ' + str(img_num) + ' Completed!\n')
        cv2.imwrite(out_dir + '/' + str(img_num) + '.bmp', original_image)
    print('\nDone!')
    print('Number of Patches: ' + str(count))
