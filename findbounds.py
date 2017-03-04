from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Flatten
import numpy as np


def bounding_nn():
    bin_pixel_width = 16
    picture_size = 256
    n_classes = np.square((picture_size / bin_pixel_width))

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
    model.fit(batch_size=batch_size)