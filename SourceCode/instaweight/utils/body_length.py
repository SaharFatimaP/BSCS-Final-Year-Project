#import cv2
import numpy as np 
import math
import time
import integeration

def body_length_depth_vec(p0,p1,d):
  
  x0,y0 = p0[0],p0[1]
  x1,y1 = p1[0],p1[1]
  NO_OF_SAMPLES = int(((x1 - x0)**2 + (y1 - y0)**2)**0.5)
  print(NO_OF_SAMPLES)

  dx = x1 - x0
  dy = y1 - y0

  # calculating 'm' that is: slope
  m = dy/dx

  # calculating 'b' that is: y-intercept.
  b = y1 - m*x1

  # interval on the x-axis
  delta_x = dx / NO_OF_SAMPLES
  x_axis = [x0 + delta_x*i for i in range(1,NO_OF_SAMPLES+1)]

  depth_vec = []
  for x in x_axis:
    # evaluating equation at: x
    y = m*x + b
    # appending the selected point in the depth_vec:
    depth_vec.append(d[int(y),int(x)])

  return depth_vec



'''data = np.load('C:\\users\\ateeb\\desktop\\depth_uint16_mm.npy')
d = data[1004] / 1000
depth_vec = body_length_depth_vec((195,185),(760,345),d)

depth_vec = np.array(depth_vec)

depth_vec[depth_vec > 4] = 0
mean = np.mean(depth_vec)
depth_vec[depth_vec == 0] = mean

depth_vec = list(depth_vec)
print(mean)
print(depth_vec)

'''
data = np.load('C:\\users\\ateeb\\desktop\\fahad_data\\1334.npy')
#x1 , y1 , x2 , y2  = 159,332 , 839,480 #1335
x1 , y1 , x2 , y2  = 836,535 , 143,331 #1334
depth_vec = body_length_depth_vec( (x1,y1) , (x2,y2) , data)

#depth_vec = data[y1 , x1:x2]
#print(depth_vec)

from matplotlib import pyplot as plt
print(repr(depth_vec))
plt.barh(np.arange(len(depth_vec)), depth_vec)
length = integeration.length_of_depth_vec(depth_vec)
print('length: ', length*39.37)
plt.show()








