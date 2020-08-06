import os
import cv2
import math
import random
import h5py as h5
import numpy as np

seed = 123
np.random.seed(seed)
random.seed(seed)

def process_dataset(path, dataset):
    print('Start to process %s ...' % (dataset))
    dir_list = sorted(os.listdir(path))
    if dataset == 'hnd':
        dir_list.remove('all.txt~')

    cnt = 0
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    for k, v in enumerate(dir_list):
        class_path = os.path.join(path, v)
        label = k

        img_list = sorted(os.listdir(class_path))
        size = len(img_list)
        test_num = math.floor(size * 0.2)
        val_num = math.floor((size - test_num) * 0.1)

        indexes = [i for i in range(size)]
        random.shuffle(indexes)
        for x, i in enumerate(indexes):
            img_path = os.path.join(class_path, img_list[i])
            img = cv2.imread(img_path, flags=0)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=0)

            if x < test_num:
                x_test.append(img)
                y_test.append(label)
            elif x < test_num + val_num:
                x_val.append(img)
                y_val.append(label)
            else:
                x_train.append(img)
                y_train.append(label)
            
            cnt += 1

            if cnt % 1000 == 0:
                print('Current step', cnt)

    print('Total number of images', cnt)
    print('x_train shape', np.array(x_train).shape)
    print('y_train shape', np.array(y_train).shape)
    print('x_val shape', np.array(x_val).shape)
    print('y_val shape', np.array(y_val).shape)
    print('x_test shape', np.array(x_test).shape)
    print('y_test shape', np.array(y_test).shape)

    f = h5.File('%s.h5' % (dataset), 'w')
    f.create_dataset('x_train', data=np.array(x_train))
    f.create_dataset('y_train', data=np.array(y_train))
    f.create_dataset('x_val', data=np.array(x_val))
    f.create_dataset('y_val', data=np.array(y_val))
    f.create_dataset('x_test', data=np.array(x_test))
    f.create_dataset('y_test', data=np.array(y_test))
    f.close()


process_dataset('English/Fnt/', 'fnt')
process_dataset('English/Hnd/Img', 'hnd')

print('Done')
