import numpy as np 
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

def tape_measure(peaks , depth_vec , found ):

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
	slicie = candidate[0]

	found.append(max_angle)
	#recursive call
	return tape_measure(peaks[slicie:] , depth_vec , found )



def smooth_using_peaks(depth_vec):
	# print("depth vec length rom smoo", len(depth_vec))
	depth_vec = np.array(depth_vec)
	peaks = find_peaks(depth_vec*-1)
	# print("len: peaks: ", len(peaks))
	# # # # # # # # # # # # # # # # # # # # # 
	peaks = list(peaks[0])
	peaks.insert(0,0)
	peaks.insert(len(peaks) , len(depth_vec)-1)
	peaks.reverse()

	print(peaks)

	found = tape_measure(peaks , depth_vec, [])
	# print(f"found : {len(found)}")
	found.insert(0 , len(depth_vec)-1)

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
	return depth_vec_new





def remove_outliers(vector):
	# mean = np.mean(vector)
	# print(mean,vector[vector>mean+0.9])
	# vector[vector>mean+0.9] = 0
	# vector[vector==0] = mean
	return vector