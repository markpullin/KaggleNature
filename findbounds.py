from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Flatten
import keras.preprocessing.image as ppi
import keras.utils.np_utils as np_utils
import numpy as np


def bounding_nn():
    path = r"C:\Users\Mark_\OneDrive\Documents\fish\train\ALB\img_00186.jpg"
    img = ppi.load_img(path, target_size=(256, 256))
    array = ppi.img_to_array(img)
    ar = np.array([array])
    bin_pixel_width = 16
    picture_size = 256
    n_classes = np.square((picture_size / bin_pixel_width))

    y = np_utils.to_categorical(1, n_classes)

    model = Sequential()
    model.add(Convolution2D(16, 3, 3, input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # add training here
    batch_size = 10
    model.fit(ar, y, nb_epoch=100)
    model.save('saved_model', overwrite=True)