
import os
import cv2

my_dir = "C:\\users\\ateeb\\desktop\\GoogleImagesResized"
for file in os.listdir("C:\\users\\ateeb\\desktop\\GoogleImagesResized"):

  image = cv2.imread(os.path.join(my_dir,file))
  cv2.imshow("Image", image)
  k= cv2.waitKey(66)

  if k == 27:
  	break
  else:
  	continue