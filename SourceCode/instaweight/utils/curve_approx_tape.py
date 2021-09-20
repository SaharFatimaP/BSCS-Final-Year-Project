import numpy as np 
from matplotlib import pyplot as plt
import integeration
import random
import math
#######################GEEKSFORGEEKS###############################
# Python code to find all three angles  
# of a triangle given coordinate  
# of all three vertices  
import math 
from scipy.signal import find_peaks  

# returns square of distance b/w two points  
def lengthSquare(X, Y):  
    xDiff = X[0] - Y[0]  
    yDiff = X[1] - Y[1]  
    return xDiff * xDiff + yDiff * yDiff 
      
def getAngle(A, B, C):  
      
    # Square of lengths be a2, b2, c2  
    a2 = lengthSquare(B, C)  
    b2 = lengthSquare(A, C)  
    c2 = lengthSquare(A, B)  
  
    # length of sides be a, b, c  
    a = math.sqrt(a2);  
    b = math.sqrt(b2);  
    c = math.sqrt(c2);  
  
    # From Cosine law  
    alpha = math.acos((b2 + c2 - a2) /
                         (2 * b * c));  
    betta = math.acos((a2 + c2 - b2) / 
                         (2 * a * c));  
    gamma = math.acos((a2 + b2 - c2) / 
                         (2 * a * b));  
  
    # Converting to degree  
    alpha = alpha * 180 / math.pi;
    beta = betta * 180 / math.pi; 
    gamma = gamma * 180 / math.pi;  
  
    # printing all the angles  
     #print("alpha : %f" %(alpha))  
    #print("betta : %f" %(betta)) 
    #print("gamma : %f" %(gamma)) 
    return alpha 

def tape_measure(peaks , depth_vec , found = [] ):

	if peaks == [] or len(peaks) == 1:
		return found

	P1 = ( depth_vec[peaks[0]] , peaks[0] )
	Pr = ( depth_vec[peaks[0]] , -1 )

	# finding the greatest angle. Candidate = (index i.e index in peaks , angle)
	candidate = (-1000, -1000)

	for i in range(1,len(peaks)):

		P2 = ( depth_vec[peaks[i]] , peaks[i] )
		angle = getAngle(P1, Pr , P2)

		if depth_vec[peaks[i]] > depth_vec[peaks[0]]:
			angle = angle * -1
		if angle > candidate[1]:
			candidate = (i , angle)

	max_angle = peaks[candidate[0]]
	slice = candidate[0]

	found.append(max_angle)
	#recursive call
	return tape_measure(peaks[slice:] , depth_vec , found )


def main(depth_vec):


	plt.barh(np.arange(len(depth_vec)), depth_vec)
	length = integeration.length_of_depth_vec(depth_vec )
	print('Arc: ',length*39.37*2)

	peaks = find_peaks(depth_vec*-1)
	# # # # # # # # # # # # # # # # # # # # # 
	peaks = list(peaks[0])
	peaks.insert(0,0)
	peaks.insert(len(peaks) , len(depth_vec)-1)
	peaks.reverse()

	for i in peaks:
		plt.plot(depth_vec[i], i, 'ro')

	found = tape_measure(peaks , depth_vec)
	found.insert(0 , len(depth_vec)-1)

	for i in found:
		plt.plot(depth_vec[i],i, 'go')




	# 3.) create lines between each split
	connections = [ (i, depth_vec[i]) for i in found]
	connections.reverse()



	line_segments = []
	for i in range(len(connections) - 1):
		rise = connections[i+1][1] - connections[i][1]
		run = connections[i+1][0] - connections[i][0]
		slope = rise/run
		line_seg = [ (connections[i][1] + k*slope) for k in range(run) ]
		line_segments.append(line_seg)

	depth_vec_new = []
	for i in line_segments:
		depth_vec_new += i

	length = integeration.length_of_depth_vec(depth_vec_new )
	print('Arc: ',length*39.37*2)
	
	plt.barh(np.arange(len(depth_vec_new)), depth_vec_new , color = 'red')
	plt.show()
 


depth = np.load('C:\\users\\ateeb\\desktop\\fahad_data\\1001.npy')
#x1 , y1 , x2 , y2  = 788 , 298 , 788 , 530 #1042
#x1 , y1 , x2 , y2  = 624,383 , 624,654 #1594
#x1 , y1 , x2 , y2  = 505,111 , 505,453 #1000.jpg
#x1 , y1 , x2 , y2  = 822,194 , 822,462 #1019
#x1 , y1 , x2 , y2  = 661,307 , 661,647 #1335
#x1 , y1 , x2 , y2  = 619,312 , 619,618 #1321
#x1 , y1 , x2 , y2  = 635,462 , 635,622 #1339
#x1 , y1 , x2 , y2  = 568,289 , 568,527 #1190
#x1 , y1 , x2 , y2  = 634,329 , 634,534 #1622
#x1 , y1 , x2 , y2  = 629,309 , 629,550 #1639
#x1 , y1 , x2 , y2  = 487,357 , 488,647 #1397
x1 , y1 , x2 , y2  = 647,184 , 647,501 #1001
#x1 , y1 , x2 , y2  = 670,379 , 670,563 #1502
#x1 , y1 , x2 , y2  = 627,458 , 627,700 #1444
#x1 , y1 , x2 , y2  = 656,329 , 656,560  #1297
depth_vec = depth[y1:y2 , x1]
'''
threshold = 2 #meters
n = np.mean(depth_vec)
for i in range(len(depth_vec)):
	if abs(depth_vec[i] - n) >= threshold:
		 depth_vec[i] = 0

m = np.mean(depth_vec)

for i in range(len(depth_vec)):
	if depth_vec[i] == 0:
		 depth_vec[i] = m

'''

main(depth_vec)