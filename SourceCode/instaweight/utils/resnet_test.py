import numpy as np 
from keras import Model
from keras.layers import Dense , GlobalAveragePooling2D
from PIL import Image, ImageDraw
from keras.applications import resnet
import numpy as np

def create_model(trainable=False):
    #model = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(IMAGE_SIZE_H,IMAGE_SIZE_W , 3), pooling='None')
    model = resnet.ResNet50(include_top=False, weights='imagenet' , input_shape = (360,640,3) )
    
    for layer in model.layers:
        layer.trainable = False

    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True
    model.layers[-5].trainable = True
    model.layers[-6].trainable = True
    model.layers[-7].trainable = True
    model.layers[-8].trainable = True
    model.layers[-9].trainable = True
    model.layers[-10].trainable = True
    model.layers[-11].trainable = True


    out = model.layers[-1].output
    x = GlobalAveragePooling2D()(out)
    x = Dense(4 , activation = 'linear')(x)

    return Model(inputs=model.input, outputs=x)



net = create_model()
net.load_weights('C:\\users\\ateeb\\desktop\\localization\\weights\\train_resnet_fullsize.hdf5')

images = np.load('C:\\users\\ateeb\\desktop\\localization\\X_cow_half_size.npy')

count = 1
for i in images:
	roi = net.predict(i.reshape(1,360,640,3))
	roi = roi[0]
	img = Image.fromarray(i)
	draw = ImageDraw.Draw(img)
	draw.rectangle( ((roi[0],roi[1]) , (roi[2],roi[3])) , outline = 'black')
	img.save('C:\\users\\ateeb\\desktop\\localization\\predictions'+'\\'+str(count)+'.jpg')
	count += 1
	print(count)
