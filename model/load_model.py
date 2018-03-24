import numpy as np
import keras.models
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,MaxPool2D

def init():
    model = Sequential()
    model.add(Conv2D(64,(3,3),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPool2D())
    model.add(Conv2D(32,(3,3),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPool2D())
    model.add(Conv2D(16,(3,3),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation = "softmax"))   
    #load woeights into new model
    model.load_weights("model.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return model, graph
