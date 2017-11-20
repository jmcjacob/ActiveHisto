import os
import cv2
import scipy.io as io
from scipy.spatial import distance


def build_binary_detection_dataset(in_dir, out_dir, patch_size, region_size, steps):
    if in_dir[-1] != '/':
        in_dir += '/'
    if out_dir[-1] != '/':
        out_dir += '/'
    border_size, count = patch_size // 2, 0
    start = int(sorted(os.listdir(in_dir))[0].replace('img', ''))
    region_count = 0
    for img_num in range(start, start + len(os.listdir(in_dir))):
        original_image = cv2.imread(in_dir + 'img' + str(img_num) + '/img' + str(img_num) + '.bmp')
        border_image = cv2.copyMakeBorder(original_image, border_size, border_size, border_size, border_size,
                                          cv2.BORDER_DEFAULT)
        points = io.loadmat(in_dir + 'img' + str(img_num) + '/img' + str(img_num) + '_detection.mat')['detection']
        for ri in range(border_size, original_image.shape[0] + border_size, region_size):
            for rj in range(border_size, original_image.shape[1] + border_size, region_size):
                if not os.path.exists(out_dir + str(region_count) + '/'):
                    os.makedirs(out_dir + str(region_count) + '/hit/')
                    os.makedirs(out_dir + str(region_count) + '/miss/')
                for pi in range(ri, ri + region_size, steps):
                    for pj in range(rj, rj + region_size, steps):
                        min_dist = 1000
                        for point in points:
                            dist = distance.euclidean((point[0], point[1]),
                                                      (pi - border_size, pj - border_size))
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist <= 3:
                            label = '/hit/'
                        elif min_dist > 20:
                            label = '/miss/'
                        else:
                            continue
                        patch = border_image[pj-border_size:pj+border_size, pi-border_size:pi+border_size]
                        patches = [patch, cv2.flip(patch, 0), cv2.flip(patch, 1), cv2.flip(patch, 2),
                                   cv2.GaussianBlur(patch, (3, 3), 0.25)]
                        for p in range(len(patches)):
                            cv2.imwrite(out_dir + str(region_count) + label + str(pi) + ',' + str(pj) + str(p) + '.bmp',
                                        patches[p])
                            count += 1
                            if count % 10000 == 0:
                                print(str(count) + ' Completed!')
                region_count += 1
                print('Region ' + str(region_count) + ' Completed!')
        print('Image ' + str(img_num) + ' Completed!')
    print('Done!')
    print('Number of Patches: ' + str(count))
