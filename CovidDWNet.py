from keras.layers import *
from keras.models import *
from keras.utils import *
import numpy as np
from tensorflow.keras.optimizers import Adam

def Future_Residual_Block(layer, nb_of_input_channels):
    x = Conv2D(nb_of_input_channels*2, kernel_size = (3,3), strides=2, padding='same', dilation_rate= (1,1), activation='relu')(layer)#kernel_size = (1,1)
    y = BatchNormalization()(x)
    for i in range(4):
        x = Conv2D(32, kernel_size = 5, strides=1, padding='same', dilation_rate= (1,1), activation='relu')(x)#kernel_size = (1,1)
        x = BatchNormalization()(x)
    x = Concatenate()([x, y])
        
    return x
    
def DepthWise_Conv(input_layer, num_channels, num_dilation, layer_name="block_Last"):
    
  x1=Future_Residual_Block(input_layer,num_channels)
  block = []
  for i in range(1, num_dilation+1):
    temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
    temp = BatchNormalization()(temp)
    block.append(temp)
        
  x = Concatenate(axis= -1)(block)
  x = Conv2D(num_channels, kernel_size = (3,3), strides=(1,1), padding='same', dilation_rate= (1,1), activation='relu',name=layer_name)(x)
  
  x = BatchNormalization()(x)


  input_tensor = x
  
  return x
  

#Network:
  
def CovidDWNet(inpt_shape, num_class):
  xin = Input(shape= inpt_shape)

  InputLayer = Conv2D(16, kernel_size = (5,5), strides= (1,1), padding = 'same', activation='relu',name = "InputLayer")(xin)
  x = BatchNormalization()(InputLayer)

  InputLayer_2 = Conv2D(32, kernel_size = (3,3), strides= (1,1), padding = 'same', activation='relu',name = "InputLayer_2")(x)#strides= (2,2)
  x = BatchNormalization()(InputLayer_2)
  
  block_1 = DepthWise_Conv(input_layer=x, num_channels=64, num_dilation=5, layer_name="block_1")
  
  
  block_2 = DepthWise_Conv(input_layer=block_1, num_channels=64, num_dilation=4, layer_name="block_2")
  
  
  block_3 = DepthWise_Conv(input_layer=block_2, num_channels=128, num_dilation=3, layer_name="block_3")


  block_4 = DepthWise_Conv(input_layer=block_3, num_channels=256, num_dilation=2, layer_name="block_Last")
  
  x = GlobalAveragePooling2D(name='Global')(block_4)

  x = Dense(128, activation='relu',name="dense_first")(x) 
  x = Dense(64, activation='relu',name="dense_two")(x)

  x = Dense(num_class, activation= 'softmax',name='predictions')(x) #
  
  model = Model(xin, x)

  model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = 1e-3), metrics = ['accuracy'])#'binary_crossentropy','categorical_crossentropy'

  return model  
