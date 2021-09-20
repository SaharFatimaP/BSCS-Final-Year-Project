import cv2
import os
import numpy as np
from PIL import Image
def scalling_factor():
	img = cv2.imread('C://users//ateeb//desktop//GoogleImages//53.jpg')
	# draw a rectangle
	cv2.rectangle(img, (579,293),(870,487), (255,0,0), 2)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# resizing the image
	img2 = cv2.resize(img , (1187//2 , 825//3) , interpolation = cv2.INTER_AREA)
	cv2.rectangle(img2, (579//2,293//3),(870//2,487//3), (0,0,255), 2)
	cv2.imshow('image', img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	'''
	Hence, it is proved, that the factor by which x-dim shrink after resizing is also the factor by which x_min shrink.
	Same is True for y-dim. Lets now rescale all image to 1280x720 !
	'''
	return None

def resizing():
	img_path = 'C://users//ateeb//desktop//GoogleImages'
	xml_path = 'C://users//ateeb//desktop//GoogleImagesAnnotations'
	images = [i for i in os.listdir(img_path)]
	bboxes = [i for i in os.listdir(xml_path)]

	# open every image and resize it:
	for i in zip(images,bboxes):
		image = i[0]
		xml = i[1]

		img = cv2.imread(img_path + '//' +image)
		h = img.shape[0]
		w = img.shape[1]
		
		h_factor = (h / 720)
		w_factor = (w / 1280)

		img = cv2.resize(img , (1280 , 720) , interpolation = cv2.INTER_AREA)

		f = open(xml_path + '//' + xml , 'r')
		data = f.readlines()
		f.close()

		xmin = int( int(data[19].split('>')[1][0:-6]) / w_factor) 
		ymin = int( int(data[20].split('>')[1][0:-6]) / h_factor) 
		xmax = int( int(data[21].split('>')[1][0:-6]) / w_factor) 
		ymax = int( int(data[22].split('>')[1][0:-6]) / h_factor) 

		data[19] = '\t\t\t<xmin>'+ str(xmin) +'</xmin>\n'
		data[20] = '\t\t\t<ymin>'+ str(ymin) +'</ymin>\n'
		data[21] = '\t\t\t<xmax>'+ str(xmax) +'</xmax>\n'
		data[22] = '\t\t\t<ymax>'+ str(ymax) +'</ymax>\n'

		#img = cv2.rectangle(img , (xmin , ymin) , (xmax, ymax) , (0,255,0) , 2)
		#cv2.imshow('image', img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		f = open('C://users//ateeb//desktop//GoogleImagesResized//'+xml , 'w')
		for i in data:
			f.write(i)
		f.close()
		cv2.imwrite('C://users//ateeb//desktop//GoogleImagesResized//'+image , img)

def resizing_bg_images():
	bg_path = 'C://users//ateeb//desktop//bg images'

	for i in os.listdir(bg_path):
		img = cv2.imread(bg_path + '//' + i)
		img = cv2.resize(img , (1280,720), interpolation = cv2.INTER_AREA)
		cv2.imwrite(bg_path	+ '//' + i , img)

	return ;

def augmentation():
	# 1) brightness
	# 2) Flip
	# 3) Apply Color filter
	img_path = 'C://users//ateeb//desktop//JPEGImages'
	xml_path = 'C://users//ateeb//desktop//Annotations_xml'
	dst_img_path = 'C://users//ateeb//desktop//dset_remaining'
	dst_xml_path = 'C://users//ateeb//desktop//dset_remaining//annotations'

	images = [ i for i in os.listdir(img_path)]
	bboxes = [ i for i in os.listdir(xml_path)]

	for i in zip(images,bboxes):

		image = i[0]
		xml = i[1]
		print(image, xml)

		img = cv2.imread(img_path + '//' +image)
		f = open(xml_path +'//'+ xml , 'r')
		data = f.readlines()
		f.close()

		cv2.imwrite(dst_img_path+'//'+image[:-4]+'.jpg' , img2)
		f = open(dst_xml_path+'//'+xml[0:-4]+'.xml','w')
		for i in data:
			f.write(i)
		f.close()

		#cv2.imshow('image', img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()		

		# brightening
		img2 = brighten(img, gamma = 2.5)
		cv2.imwrite(dst_img_path+'//'+image[:-4]+'_b.jpg' , img2)
		f = open(dst_xml_path+'//'+xml[0:-4]+'_b.xml','w')
		
		for i in data:
			f.write(i)
		f.close()
		
		#cv2.imshow('image', img2)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		# adding noise
		img3 = brighten(img , gamma = 0.4)
		img3 = noise('gauss' , img)
		cv2.imwrite(dst_img_path+'//'+image[:-4]+'_bn.jpg' , img3)
		f = open(dst_xml_path+'//'+xml[0:-4]+'_bn.xml','w')
		for i in data:
			f.write(i)
		f.close()

		#cv2.imshow('image', img3)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		
		#flipping (special case)
		
		img4 = flip(img)
		img4 = noise('gauss' , img4 , var = 160)

		xmin = int(data[31].split('>')[1][0:-6]) 
		ymin = int(data[32].split('>')[1][0:-6]) 
		xmax = int(data[33].split('>')[1][0:-6]) 
		ymax = int(data[34].split('>')[1][0:-6])
		xmin , xmax = 1280 - xmax , 1280 - xmin		

		#img4 = cv2.circle(img4 , (xmin,ymin) ,3, (255,0,0) ,5)
		#img4 = cv2.circle(img4 , (xmax,ymax) ,3, (0,0,255) ,5)
		#img4 = cv2.rectangle(img4,(xmin,ymin),(xmax,ymax) , (0,255,0) , 2)
		#cv2.imshow('image', img4)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		data[31] = '\t\t\t<xmin>'+ str(xmin) +'</xmin>\n'
		data[32] = '\t\t\t<ymin>'+ str(ymin) +'</ymin>\n'
		data[33] = '\t\t\t<xmax>'+ str(xmax) +'</xmax>\n'
		data[34] = '\t\t\t<ymax>'+ str(ymax) +'</ymax>\n'
		
		cv2.imwrite(dst_img_path+'//'+image[:-4]+'_f.jpg' , img4)
		f = open(dst_xml_path+'//'+xml[0:-4]+'_f.xml','w')
		for i in data:
			f.write(i)
		f.close()


def noise(noise_type , image , var = 250):
	if noise_type == 'gauss':
		row, col , ch = image.shape
		mean = 0
		sigma = var ** 0.5
		gauss = np.random.normal(mean , sigma , (row,col,ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = image + gauss
		cv2.normalize(noisy , noisy , 0,255 , cv2.NORM_MINMAX , dtype = -1)
		noisy = noisy.astype(np.uint8)
		return noisy

def brighten(image , gamma = 0.5):
	invGamma = 1.0/gamma 
	table = np.array( [((i/255.0) ** invGamma)* 255 for i in np.arange(0,256)] ).astype('uint8')
	return cv2.LUT(image, table)

def flip(image):
	image = cv2.flip(image , 1)
	return image


def inner_crop():
	img_path = "C:\\users\\ateeb\\desktop\\compiled dataset\\images"
	xml_path = "C:\\users\\ateeb\\desktop\\compiled dataset\\annotations"

	images = [ i for i in os.listdir(img_path)]
	bboxes = [ i for i in os.listdir(xml_path)]

	X_cow = []
	y_cow = []

	count = 0
	for i in zip(images , bboxes):
		image = i[0]
		xml = i[1]

		#step_1: load the image
		img = cv2.imread(img_path + '\\' + image)
		#converting from BGR to RGB
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		#step_2: load the xml
		f = open(xml_path + '\\' + xml , 'r')
		data = f.readlines()
		f.close()

		xmin = int(data[19].split('>')[1][0:-6])
		ymin = int(data[20].split('>')[1][0:-6])
		xmax = int(data[21].split('>')[1][0:-6])
		ymax = int(data[22].split('>')[1][0:-6])

		xmin_inner = int(data[31].split('>')[1][0:-6])
		ymin_inner = int(data[32].split('>')[1][0:-6])
		xmax_inner = int(data[33].split('>')[1][0:-6])
		ymax_inner = int(data[34].split('>')[1][0:-6])

		#step_3: Crop the cow only from the image
		img = img[ymin:ymax , xmin:xmax]
		#step_4: Since the image is cropped we have to adjust the values of inner crop accordingly
		xmin_inner -= xmin
		xmax_inner -= xmin
		ymin_inner -= ymin
		ymax_inner -= ymin

		#step_5: we resize the cropped image to, note: The aspect ratio is 4:3 and resolution is 400x300
		prev_shape = img.shape
		img = cv2.resize(img , (400,300))
		#step_6: now we rescale the image
		h = img.shape[0]
		w = img.shape[1]
		h_factor = (h / prev_shape[0])
		w_factor = (w / prev_shape[1])

		xmin_inner = int(xmin_inner * w_factor)
		xmax_inner = int(xmax_inner * w_factor)
		ymin_inner = int(ymin_inner * h_factor)
		ymax_inner = int(ymax_inner * h_factor)

		# for sanity check
		'''img = cv2.rectangle(img , (xmin_inner, ymin_inner) , (xmax_inner , ymax_inner) , (25,55,213) , 3)
		p = Image.fromarray(img)
		p.show()
		if count == 5:
			break
		count += 1
		continue'''


		#step_4: Making it a numpy array
		img = np.asarray(img)
		X_cow.append(img)
		y_cow.append([xmin_inner, ymin_inner , xmax_inner , ymax_inner])
		print(count , img.shape , image , xml)
		# increasing the counter
		count +=1

	# saving the X_cow and Y_cow as np array
	np.save('X_cow_depth_data_inner.npy', X_cow)
	np.save('y_cow_depth_data_inner.npy', y_cow)


		#step_5: Calculate the new bounding box







def Xy_creator():
	from PIL import Image 
	img_path = "C:\\users\\ateeb\\desktop\\compiled dataset\\images"
	xml_path = "C:\\users\\ateeb\\desktop\\compiled dataset\\annotations"

	images = [ i for i in os.listdir(img_path)]
	bboxes = [ i for i in os.listdir(xml_path)]



	X_cow = []
	y_cow = []
	count = 0
	for i in zip(images , bboxes):
		print(count)
		image = i[0]
		xml = i[1]

		img = Image.open(img_path + '\\' + image)
		# resizing the image by half
		img = img.resize((640,360), Image.ANTIALIAS)
		img = np.array(img)
		X_cow.append(img)

		f = open(xml_path + '\\' + xml , 'r')
		#print(xml)
		data = f.readlines()
		f.close()

		#resizing the ROI by half, as we have resized the image to half as well.
		xmin = int(data[19].split('>')[1][0:-6]) // 2
		ymin = int(data[20].split('>')[1][0:-6]) // 2
		xmax = int(data[21].split('>')[1][0:-6]) // 2
		ymax = int(data[22].split('>')[1][0:-6]) // 2	

		y = np.array([xmin , ymin , xmax, ymax])
		y_cow.append(y)
		count += 1
		
	np.save('X_cow_depth_data.npy', X_cow)
	np.save('y_cow_depth_data.npy', y_cow)


#resizing()
#resizing_bg_images()
#augmentation()
#Xy_creator()
inner_crop()