import numpy as np
from PIL import Image
import os

def view(im , i = 'E:\\1200f.jpg'):
	im = im / np.max(im)
	im = (im * 255) + 100
	img = Image.fromarray(im)
	img = img.convert("L")
	img.save(i)

def zero_filter(d):
	for i in range(len(d)):
		for j in range(len(d[i])):

			if d[i][j] == 0:
				
				bottom = 0
				left = 0
				right = 0
				top = 0

				try:
					bottom = (1/4)*d[i+1][j]
				except:
					pass
				try:
					top = (1/4)*d[i-1][j]
				except:
					pass
				try:
					right = (1/4)*d[i][j+1]
				except:
					pass
				try:
					left = (1/4)*d[i][j-1]
				except:
					pass

				d[i][j] = bottom + top + left + right
	
	for i in range(len(d)):
		for j in range(len(d[i])):
			if d[i][j] >= 3:
				d[i][j] = 0

	return d


for i in os.listdir('E:\\depth_dataset_backup\\'):
	if i[-3:] == 'npy':
		d = np.load('E:\\depth_dataset_backup\\' + i)
		d = zero_filter(d)
		np.save('E:\\depth filtered\\'+i , d)

