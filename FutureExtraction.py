
# plot feature map of first conv layer for given image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

#block_1, InputLayer, InputLayer_2

def plot_show(img_path,model):
    
    # redefine model to output right after the first hidden layer
    ixs = [1,2,27, 49, 69, 87]
    outputs = [model.layers[i].output for i in ixs]
    model_Future = Model(inputs=model.inputs, outputs=outputs)
    #load the image with the required shape
    img = load_img(img_path, target_size=(128, 128))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model_Future.predict(img)
    # plot the output from each block
    square =4
    
    ly=1
    
    for fmap in feature_maps:
        ix=1
        #print("fmap")
        #print(fmap.shape)
        print(ly)
        print("")
        print("")
        ly+=1
        for _ in range(square):
          for _ in range(square):
            ax=pyplot.subplot(square,square,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
        
        pyplot.show()


