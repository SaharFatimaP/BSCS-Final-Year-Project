import numpy as np 
import os
from PIL import Image



img_path = 'C:\\users\\ateeb\\desktop\\compiled dataset\\Dataset Ateeb\\'

for i in os.listdir(img_path):
	print(i)
	img = Image.open(img_path + i)
	img = img.resize((1000 , 750) , resample = Image.LANCZOS)
	
	#saving the resize image
	img.save(img_path + i)




 
