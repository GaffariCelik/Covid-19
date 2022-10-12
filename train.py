from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import *
from CovidDWNet import CovidDWNet

def data_load(data_path):
    

    
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale = 1./255)
    
    
    train_generator = train_datagen.flow_from_directory(data_path+'/train',
                                         target_size=(128, 128), 
                                         batch_size = 32, 
                                         shuffle = True,class_mode='categorical')
    
    val_generator = val_datagen.flow_from_directory(data_path+'/test',
                                     target_size=(128, 128),
                                     batch_size =32,
                                     shuffle =False,class_mode='categorical')
    return train_generator,val_generator                                     
                                 

def train(data_path):
    BS=32
    EPOCHS=200    
    
    train_generator,val_generator=data_load(data_path)
    model = CovidDWNet(inpt_shape = (128, 128, 3), num_class = 4)
    #model.summary() 
    
    fname="checkpoint/our_model.h5" 
    callbacks = ModelCheckpoint(fname, monitor="val_accuracy", mode="max",
                                  save_best_only=True, verbose=1)#,save_freq=50*(train_generator.samples//BS))
    callbacks=[callbacks]
    
    H = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//BS,
        validation_data=val_generator,
        validation_steps=val_generator.samples//BS,
        epochs=200,
        initial_epoch=0,
        verbose=1,callbacks=callbacks)



if __name__ == '__main__':
    data_path='data'
    train(data_path)