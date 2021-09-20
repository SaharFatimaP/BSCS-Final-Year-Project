from PIL import Image
import numpy as np 
import os

image_path = 'E:\\fyp work\\localization\\cattle.jpg'
mask_path = 'E:\\fyp work\\masks'


for i in os.listdir(mask_path):
	mask = np.load(mask_path + '\\' + i)
	#image = Image.open(image_path)
	#image = image.convert('L')
	#image = np.array(image)

	mask = mask*255
	mask = Image.fromarray(mask)
	mask.save('E:\\masks cattle' + '\\' + i[0:-4] + '.jpg' )


#image.show()
#for i in range(len(mask)):
#	for j in range(len(mask[i])):
#		if mask[i][j] == 1:
#			image[i][j] = 0

#image = Image.fromarray(image)
#image.save('E:\\mask_bw.jpg')
