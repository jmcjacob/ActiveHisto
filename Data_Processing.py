import os
import cv2
import sys
import numpy as np
import scipy.io as io
from scipy.spatial import distance


def is_plain(patch):
    grey_image = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if np.average(grey_image) > 220:
        return True


def build_dataset(in_dir, out_dir, patch_size, region_size, steps, scale, negative_augmentation=True):
    if in_dir[-1] != '/':
        in_dir += '/'
    if out_dir[-1] != '/':
        out_dir += '/'
    border_size, count = patch_size // 2, 0
    start = int(sorted(os.listdir(in_dir))[0].replace('img', ''))
    region_count = 0
    for img_num in range(start, start + len(os.listdir(in_dir))):
        image = cv2.imread(in_dir + 'img' + str(img_num) + '/img' + str(img_num) + '.bmp')
        image = cv2.resize(image, (0,0), fx=scale, fy=scale)
        border_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_DEFAULT)
        points = io.loadmat(in_dir + 'img' + str(img_num) + '/img' + str(img_num) + '_detection.mat')['detection']
        for ri in range(border_size, image.shape[0] + border_size, region_size):
            for rj in range(border_size, image.shape[1] + border_size, region_size):
                if not os.path.exists(out_dir + str(region_count) + '/'):
                    os.makedirs(out_dir + str(region_count) + '/positive/')
                    os.makedirs(out_dir + str(region_count) + '/negative/')
                for pi in range(ri, ri + region_size, steps):
                    for pj in range(rj, rj + region_size, steps):
                        min_dist = 1000
                        for point in points:
                            dist = distance.euclidean((point[0] * scale, point[1] * scale), (pi - border_size, pj - border_size))
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist <= image.shape[0] // 100:
                            label = '/positive/'
                        elif min_dist > ((image.shape[0] // 10) / 5) * 2:
                            label = '/negative/'
                        else:
                            continue
                        patch = border_image[pj-border_size:pj+border_size + 1, pi-border_size:pi+border_size + 1]
                        if is_plain(patch):
                            continue
                        if label == '/positive/':
                            patches = [patch, cv2.flip(patch, 0), cv2.flip(patch, 1), cv2.flip(patch, 2),
                                       cv2.GaussianBlur(patch, (3, 3), 0.25)]
                        elif negative_augmentation:
                            patches = [patch, cv2.flip(patch, 0), cv2.flip(patch, 1), cv2.flip(patch, 2),
                                       cv2.GaussianBlur(patch, (3, 3), 0.25)]
                        else:
                            patches = [patch]
                        for p in range(len(patches)):
                            cv2.imwrite(out_dir + str(region_count) + label + str(pi) + ',' + str(pj) + str(p) + '.bmp',
                                        patches[p])
                            count += 1
                            if count % 10 == 0:
                                print(str(count) + ' Completed!')
                region_count += 1
                print('Region ' + str(region_count) + ' Completed!')
        print('Image ' + str(img_num) + ' Completed!')
    print('Done!')
    print('Number of Patches: ' + str(count))


if __name__ == '__main__':
    if str(sys.argv[7]).lower() == 'true':
        augment_neg = True
    else:
        augment_neg = False
    build_dataset(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]),
                  augment_neg)
