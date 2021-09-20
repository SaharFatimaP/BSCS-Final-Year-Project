from PIL import Image
import numpy as np
import os

#data = np.load('C:\\users\\ateeb\\desktop\\fahad_data\\1339.npy')

def view(im , i):
	im = im / np.max(im)
	im = im * 255
	im = im + 100
	img = Image.fromarray(im)
	img = img.convert("L")
	img.save(i)



path = 'C:\\users\\ateeb\\desktop\\test images\\depths'
save_path = path
files = os.listdir(path)
for i in files:
	if i != '1221.npy':
		continue
	if i[-4:] == '.npy':
		print(i)
		depth = np.load(path+'\\'+i)
		i = i[:-4]+'.jpg'
		view(depth , save_path+'\\'+i)
