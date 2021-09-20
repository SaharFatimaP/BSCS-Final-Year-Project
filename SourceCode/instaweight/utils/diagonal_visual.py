import cv2
from PIL import Image
import numpy as np 

for i in diag:
	x = i[0]
	y = i[1]

	img = cv2.circle(img , (x,y) , 1 , (0,255,0), 1)
	cv2.imshow('image',img)
	cv2.waitKey(1)


cv2.waitKey(0)
cv2.destroyAllWindows()
