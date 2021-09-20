import glob
import cv2
import numpy as np
from model import create_model
from keras.applications import vgg16 

IMAGE_SIZE_W = 1280//2
IMAGE_SIZE_H = 720//2

WEIGHTS_FILE = "C:/users/ateeb/desktop/train_vgg16_cwh_N.hdf5"
IMAGES = "./test_images/*jpg"


#IMAGES = "C:/Users/folio3/Downloads/dset/dset/*jpg"

'''model = create_model(
    IMAGE_SIZE_H, 
    IMAGE_SIZE_W,
    '../Weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )
'''
model_half = create_model()
model_half.load_weights("C:/users/ateeb/desktop/train_vgg16_cwh_N_full.hdf5")
print('fuck')
#print(model_half.summary())

for filename in glob.glob(IMAGES):
    print(filename)
    image = cv2.imread(filename)
    image = cv2.resize( image , (640,360) , interpolation = cv2.INTER_AREA)

    #dim = (IMAGE_SIZE_W, IMAGE_SIZE_H)
    #image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #region = model_half.predict(x=np.array([image]))[0]
    region = model_half.predict(x=np.array([image]))[0]
    print(region)
    image_height, image_width, _ = image.shape
    

    x = int(region[0]*640)
    y = int(region[1]*360)

    w = int(region[2]*640)
    h = int(region[3]*360)

    cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 255), 2)
    print((x - w//2, y - h//2), (x + w//2, y + h//2))
    #cv2.rectangle(image, (x2, y2), (x3, y3), (0, 100, 255), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

