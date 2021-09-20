import numpy as np
from keras import Model,  optimizers
from keras.applications import resnet , vgg16
from keras.layers import Conv2D, Reshape, GlobalAveragePooling2D, Flatten, Dense , Dropout , MaxPool2D
from keras.utils import Sequence
from keras.backend import epsilon
from keras.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw


def create_model(trainable=False):
    #model = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(IMAGE_SIZE_H,IMAGE_SIZE_W , 3), pooling='None')
    model = resnet.ResNet50(include_top = False,weights=None , input_shape = (300,400,3) )
    
    for layer in model.layers:
        layer.trainable = False

    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True

    out = model.layers[-1].output
    x = GlobalAveragePooling2D()(out)
    x = Dense(4 , activation = 'linear')(x)

    return Model(inputs=model.input, outputs=x)

def create_model_vgg(trainable=False):
    model = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(300,400,3), pooling='None')
    #model = resnet.ResNet50(include_top = False,weights=None , input_shape = (300,400,3) )
    
    for layer in model.layers:
        layer.trainable = False




    out = model.layers[-1].output
    x = GlobalAveragePooling2D()(out)
    x = Dense(512 , activation='relu')(x)
    x = Dense(4 , activation = 'linear')(x)

    return Model(inputs=model.input, outputs=x)

#loading the dataset
model = create_model_vgg()
model.load_weights('C:\\users\\ateeb\\desktop\\localization\\weights\\inner_vgg16_cropped_images_8ld_GAP_512.hdf5')

img = Image.open('C:\\users\\ateeb\\desktop\\test images\\875.jpg')
#img = img.resize((400,300), Image.ANTIALIAS)
a = np.asarray(img)
r = model.predict(x=np.array([a]))[0]
print(r)
draw = ImageDraw.Draw(img)
draw.rectangle(((r[0],r[1]),(r[2],r[3])) , outline = 'blue')
img.show()
img.save('888_p.jpg')
#loading the dataset
#X_test = np.load('C:\\users\\ateeb\\desktop\\localization\\X_cow_depth_data_inner.npy')[0:10]
#y_test = np.load('C:\\users\\ateeb\\desktop\\localization\\y_cow_depth_data_inner.npy')[0:10]

'''#predictions
from PIL import Image , ImageDraw
for i in X_test:
    region = model.predict(x=np.array([i]))[0]
    img = Image.fromarray(i)
    draw = ImageDraw.Draw(img)
    x = region[0] 
    y = region[1]
    x1 = region[2]
    y1 = region[3]
    print(region)

    draw.rectangle( ( (x,y) , (x1,y1) ) , fill="red")
    img.show()'''