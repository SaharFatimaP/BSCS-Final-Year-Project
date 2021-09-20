import numpy as np
import cv2



img = cv2.imread('C:\\users\\ateeb\\desktop\\155.jpg')
depth = np.load('C:\\users\\ateeb\\desktop\\155.npy')


# VISAULIZING THE ROI
f = open('C:\\users\\ateeb\\desktop\\155.xml', 'r')
data = f.readlines()
f.close()

xmin =  int(data[19].split('>')[1][0:-6]) 
ymin =  int(data[20].split('>')[1][0:-6])  
xmax =  int(data[21].split('>')[1][0:-6])  
ymax =  int(data[22].split('>')[1][0:-6])  

print(xmin , ymin , xmax , ymax)
img = cv2.rectangle(img , (xmin,ymin),(xmax,ymax), (0,225,0) ,3)


# ANALYSING THE DEPTH INFORMATION

roi = depth[ymin:ymax , xmin:xmax]
mean_depth_roi = np.mean(roi)
var_depth_roi = np.var(roi)
max_depth_roi = np.unravel_index(roi.argmax(), roi.shape)
min_depth_roi = np.unravel_index(roi.argmin(), roi.shape)


# VISUALIZING THE points with MIN, MAX depth on the image
c_c = (max_depth_roi[0] + ymin , max_depth_roi[1] + xmin)
print( depth[c_c[0]][c_c[1]] )

img = cv2.circle(img, (c_c[1],c_c[0]), 5 , (0,255,0), 30)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()