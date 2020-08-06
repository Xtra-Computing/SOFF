import os
import cv2
import csv
import h5py as h5
import numpy as np

if not os.path.isdir('data'):
    os.mkdir('data')

def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
    img_flip = cv2.flip(img, 1)
    
    img = np.transpose(img, (2, 0, 1))
    img_flip = np.transpose(img_flip, (2, 0, 1))
    
    return [img, img_flip]


def process_dataset(dataset):
    print('Start to process rfw %s ...' % (dataset))

    label_path = 'test/txts/{0}/{0}_pairs.txt'.format(dataset)
    
    csvFile = open(label_path, 'r')
    reader = csv.reader(csvFile, delimiter='\t')
    cnt = 0

    img1 = []
    img1_flip = []
    img2 = []
    img2_flip = []
    labels = []
    for row in reader:
        length = len(row)

        img1_path = 'test/data/{0}/{1}/{1}_{2}.jpg'.format(dataset, row[0], row[1].zfill(4))
        img1s = process_image(img1_path)
        img1.append(img1s[0])
        img1_flip.append(img1s[1])

        if length == 3:
            img2_path = 'test/data/{0}/{1}/{1}_{2}.jpg'.format(dataset, row[0], row[2].zfill(4))
            labels.append([1])
        elif length == 4:
            img2_path = 'test/data/{0}/{1}/{1}_{2}.jpg'.format(dataset, row[2], row[3].zfill(4))
            labels.append([0])

        img2s = process_image(img2_path)
        img2.append(img2s[0])
        img2_flip.append(img2s[1])

        cnt += 1
        if cnt % 1000 == 0:
            print('Current step', cnt)


    print('Total number of images', cnt)
    print('img1 shape', np.array(img1).shape)
    print('img1_flip shape', np.array(img1_flip).shape)
    print('img2 shape', np.array(img2).shape)
    print('img2_flip shape', np.array(img2_flip).shape)
    print('labels shape', np.array(labels).shape)

    f = h5.File('data/%s_test.h5' % (dataset), 'w')
    f.create_dataset('img1', data=np.array(img1))
    f.create_dataset('img1_flip', data=np.array(img1_flip))
    f.create_dataset('img2', data=np.array(img2))
    f.create_dataset('img2_flip', data=np.array(img2_flip))
    f.create_dataset('labels', data=np.array(labels))
    f.close()


process_dataset('Caucasian')
process_dataset('African')

print('Done')

