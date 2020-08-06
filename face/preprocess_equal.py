import os
import cv2
import random
import h5py as h5
import numpy as np

seed = 123
np.random.seed(seed)
random.seed(seed)

if not os.path.isdir('data'):
    os.mkdir('data')

def process_dataset(dataset):
    print('Start to process %s ...' % (dataset))
    path = 'race_per_7000/%s/' % (dataset)
    dir_list = sorted(os.listdir(path))

    cnt = 0
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for k, v in enumerate(dir_list):
        class_path = os.path.join(path, v)
        label = k

        img_list = sorted(os.listdir(class_path))
        val_index = random.randint(0, len(img_list)-1)

        for x, i in enumerate(img_list):
            img_path = os.path.join(class_path, i)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
            img = np.transpose(img, (2, 0, 1))

            if x == val_index:
                x_val.append(img)
                y_val.append(label)
            else:
                x_train.append(img)
                y_train.append(label)
            
            cnt += 1

            if cnt % 10000 == 0:
                print('Current step', cnt)

    print('Total number of images', cnt)
    print('x_train shape', np.array(x_train).shape)
    print('y_train shape', np.array(y_train).shape)
    print('x_val shape', np.array(x_val).shape)
    print('y_val shape', np.array(y_val).shape)

    f = h5.File('data/%s_train_val.h5' % (dataset), 'w')
    f.create_dataset('x_train', data=np.array(x_train))
    f.create_dataset('y_train', data=np.array(y_train))
    f.create_dataset('x_val', data=np.array(x_val))
    f.create_dataset('y_val', data=np.array(y_val))
    f.close()


process_dataset('Caucasian')
process_dataset('African')

print('Done')


