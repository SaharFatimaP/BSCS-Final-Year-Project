#ateeb
import cv2
import numpy as np 
import math
import time

# NO_OF_SAMPLES should be propotional to the difference of two given points P1 and P2.
img = cv2.imread('C:\\users\\ateeb\\desktop\\brehensen.png')
depth_vec = []

x1,y1 = 160,145
x0,y0 = 603,296

NO_OF_SAMPLES = int(((x1 - x0)**2 + (y1 - y0)**2)**0.5)
print(NO_OF_SAMPLES)
dx = x1 - x0
dy = y1 - y0
print(dx , dy)

# get the equation of the form: y = mx + b
# calculating 'b' that is: y-intercept.
b = y1 - (dy/dx)*x1

# calculating 'm' that is: slope
m = dy/dx

# interval on the x-axis
delta_x = dx / NO_OF_SAMPLES
x_axis = [x0 + delta_x*i for i in range(1,NO_OF_SAMPLES+1)]

for x in x_axis:
	# evaluating equation at: x
	y = m*x + b
	depth_vec.append((int(x) , int(y)))
	cv2.circle(img , (int(x),int(y)) , 1 , (0,255,0), 1)
	cv2.imshow("Image",img)
	cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(len(depth_vec))
print(depth_vec)

'''
# calculating the new equation which is of the form : f(x,y) = Ax + By + C = 0
# where A = dy , B = -dx , C = (dx)(b)
A = dy
B = -dx
C = (dx)*(b)

x,y = x0,y0
depth_vec = [(x,y)]

while(True):
	# candidate point to be evaluated:
	mid_x , mid_y = x + 1 , y + 0.5

	#evaluating the line at (mid_x , mid_y):
	e = A*mid_x + B*mid_y + C

	# selecting the next point:
	if e >= 0:
		x , y = x+1 , y+1
	elif e < 0:
		x , y = x+1 , y

	# appending the selected point in the depth_vec:
	depth_vec.append((x,y))
	cv2.circle(img , (x,y) , 1 , (0,255,0), 1)
	cv2.imshow("Image",img)
	cv2.waitKey(2)
	
	# checking the stopping condition:
	if x == x1 and y == y1:#) or (x== x1+1 and y == y1+1):
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		break
'''