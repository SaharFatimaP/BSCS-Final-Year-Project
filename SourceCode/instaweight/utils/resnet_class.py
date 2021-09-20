from keras.applications.resnet50 import ResNet50
from keras.models import Model   
from keras.layers import Dense


model = ResNet50(include_top=False, weights=None, input_shape=(360,640 , 3), pooling='avg')
for layer in model.layers[:-10]:
  layer.trainable = False 
#model.summary ()

out = model.layers[-1].output
x = Dense(2, activation="softmax")(out)

classification_model = Model(inputs=model.input, outputs=x)
