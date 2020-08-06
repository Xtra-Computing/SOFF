# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import csv

read_path = 'dirc/to/read/data'
write_path = 'dirc/to/write/data'
error_path = 'error_img.csv'

if not os.path.isdir(write_path):
    os.makedirs(write_path)

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=9, accurate_landmark=False)

cnt = 0
error_img = []
dirc_list = os.listdir(read_path)

start = time.time()
for i, dirc in enumerate(dirc_list):
    dirc_path = read_path + dirc
    dirc_path_new = write_path + dirc
    os.makedirs(dirc_path_new)
    file_list = os.listdir(dirc_path)

    for file in file_list:
        file_path = dirc_path + '/' + file
        img = cv2.imread(file_path)

        # run detector
        results = detector.detect_face(img)
        if results is None:
            print('Warning:', file_path)
            error_img.append([file_path])
            img = cv2.resize(img, (112, 112))
            cv2.imwrite(dirc_path_new+'/'+file, img)
        else:
            total_boxes = results[0]
            points = results[1]
            index = 0

            if len(total_boxes) > 1:
            	area = []
            	for box in total_boxes:
            		area.append((box[2] - box[0]) * (box[3] - box[1]))
            	index = area.index(max(area))
            
            # extract aligned face chips
            chip = detector.extract_image_chips(img, [points[index]], 112, 0.37)
            cv2.imwrite(dirc_path_new+'/'+file, chip[0])

        cnt += 1
        if cnt % 1000 == 0:
            print('Current step', cnt)


end = time.time()
print(end - start)

with open(error_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(error_img)

print('error img num', len(error_img))
print('total img num', cnt)
print('Done')
