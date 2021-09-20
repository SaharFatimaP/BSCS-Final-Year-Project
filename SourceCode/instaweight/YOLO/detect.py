import image_detect
import cv2
yolo = image_detect.YOLO()


while(True):
	
	image = input('Enter path to image:')
	r_image , ObjectList = yolo.detect_img(image)
	
	print(ObjectList)
	
	cv2.imshow('detected image' , r_image)
	
	if cv2.waitKey(25) & 0xFF == ord("q"):
		cv2.destroyAllWindows()

yolo.close_session()

