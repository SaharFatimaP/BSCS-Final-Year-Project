from keras import Model
from keras.applications import resnet
from keras.layers import Dense, GlobalAveragePooling2D




def create_model(trainable=False):
    model = resnet.ResNet50(include_top=False, weights='imagenet' , input_shape = (360 , 640 ,3 ) , pooling = 'None')
    for layer in model.layers:
        layer.trainable = False
    
    out = model.layers[-1].output
    x = GlobalAveragePooling2D()(out)
    x = Dense(4 , activation = 'sigmoid')(x)

    return Model(inputs=model.input, outputs=x)