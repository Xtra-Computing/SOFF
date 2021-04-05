import os
import cv2
import math
import random
import h5py as h5
import numpy as np

def scan_data(path):
	print(path)
	dir_list = sorted(os.listdir(path))
	print('dir len', len(dir_list))

	cnt = 0
	cnt2 = 0
	for k, v in enumerate(dir_list):
		class_path = os.path.join(path, v)
		img_list = sorted(os.listdir(class_path))
		for i in img_list:
			if 'align' in i:
				cnt2 += 1
			cnt += 1

	print('Total and aligned images number', cnt, cnt2)

scan_data('race_per_7000/Caucasian/')
scan_data('race_per_7000/African/')
scan_data('race_per_7000/Asian/')
scan_data('race_per_7000/Indian/')

