
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import *
import numpy as np


def load_data(path):    
    BS=32
    EPOCHS=200
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale = 1./255)
    

    
    
    train_generator = train_datagen.flow_from_directory(path+'/train',
                                         target_size=(128, 128), 
                                         batch_size = BS, 
                                         shuffle = True,class_mode='categorical')
    
    val_generator = val_datagen.flow_from_directory(path+'/test',
                                     target_size=(128, 128),
                                     batch_size =BS,
                                     shuffle =False,class_mode='categorical')
    return train_generator,val_generator
                                 

def load_test(val_generator):
    
    test_set=val_generator
    test_set.reset()
    testX, testY = next(test_set)
    BS=32
    for i in range(test_set.samples//BS):
      img, label = next(test_set)
      testX = np.append(testX, img, axis=0 )
      testY = np.append(testY, label, axis=0)
    
    return testX,testY      

def load_train(train_generator):
    train_set=train_generator
    train_set.reset()
    trainX, trainY = next(train_set)
    BS=32
    
    for i in range(train_set.samples//BS): 
      img, label = next(train_set)
      trainX = np.append(trainX, img, axis=0 )
      trainY = np.append(trainY, label, axis=0)
    
    return trainX,trainY        
  
  
def create_Vektor(model,data_path):
    train_generator,val_generator=load_data(data_path)
    testX,testY=load_test(val_generator)
    trainX,trainY=load_train(train_generator)
    

    vector_layers = ['dense_first', 'dense_two']
    model_for_vector = Model(
        
        inputs=model.input,
        outputs=model.get_layer('dense_two').output#get_layer(index=90).output
    )

    v_X_train = model_for_vector.predict(trainX)
    v_X_train = np.array(v_X_train)

    v_X_test = model_for_vector.predict(testX)
    v_X_test = np.array(v_X_test)
    
    return v_X_train, v_X_test, trainY, testY

 
