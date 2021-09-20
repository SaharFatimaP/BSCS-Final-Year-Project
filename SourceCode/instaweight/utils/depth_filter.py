import os
import numpy as np 
from PIL import Image , ImageDraw




path = 'E://depth_dataset_backup'
d_threshold = 1.2


def draw_rectangle(img , w, h, window_size):
	draw = ImageDraw.Draw(img)
	x1 , y1 , x2 , y2 = w-window_size , h-window_size , w+window_size , h+window_size
	draw.rectangle( ((x1,y1) , (x2,y2)) , outline = 'green')
	return img

def image_save(im , i , h, w ,window_size):
	im = im / np.mean(im)
	im = im * 255
	img = Image.fromarray(im)
	img = img.convert("L")

	img = draw_rectangle(img , w , h , window_size)
	img.save(i)


def contains_zero(window):
	for i in window:
		for j in i:
			if j == 0:
				return True
	return False

def filter(threshold , path , window_size = 5):
	for i in os.listdir(path):
		# if jpeg ignore and reiterate
		if i[-3:] == 'jpg':
			continue
		print(i)
		depth = np.load(path + '//' + i)
		w = depth.shape[1] // 2 
		h = depth.shape[0] // 2 

		window = depth[h-window_size:h+window_size , w-window_size:w+window_size]
		
		if contains_zero(window):
			# save the image
			image_save(depth ,'E:\\filter\\bad\\' + i[:-4]+ '.jpg' , h ,w , window_size)

		else:
			image_save(depth , 'E:\\filter\\good\\'+ i[:-4]+ '.jpg' , h , w , window_size)


filter(d_threshold , path)