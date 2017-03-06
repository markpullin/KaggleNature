import os

import numpy as np
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
import PIL.Image as Image

import preprocessing
from preprocessing import scale_image


def subsample_from_no_fish_pictures():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    folder = os.path.join(dir_path, r"train\NoF")
    files = os.listdir(folder)
    images = []
    for file in files:
        img = Image.open(os.path.join(folder, file))
        x_lower = int(np.ceil(np.random.rand(1) * (img.size[0] - 256)))
        y_lower = int(np.ceil(np.random.rand(1) * (img.size[1] - 256)))
        cropped = img.crop((x_lower, y_lower, x_lower + 256, y_lower + 256))
        images.append(np.asarray(cropped))
    return images


def create_cropped_images_of_fish(points):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_folder = os.path.join(dir_path, r"train\ALB")
    file_names = points.keys()

    cropped_images = []
    for file_name in file_names:
        this_points = points[file_name]
        if len(this_points) == 2:
            img = Image.open(os.path.join(root_folder, file_name))
            x_br = np.max((np.min((this_points[0][0], this_points[1][0])) - 50, 0))
            y_br = np.max((np.min((this_points[0][1], this_points[1][1])) - 50, 0))
            x_tl = np.min((np.max((this_points[0][0], this_points[1][0])) + 50, img.size[0]))
            y_tl = np.min((np.max((this_points[0][1], this_points[1][1])) + 50, img.size[1]))
            # make this into a square!
            x_length = x_tl - x_br
            y_length = y_tl - y_br
            if x_length > y_length:
                y_tl += (x_length - y_length) / 2
                y_br -= (x_length - y_length) / 2
            else:
                x_tl += (y_length - x_length) / 2
                x_br -= (y_length - x_length) / 2

            cropped_img = img.crop((x_br, y_br, x_tl, y_tl))
            cropped_img = cropped_img.resize((256, 256))
            cropped_images.append(np.asarray(cropped_img))
    return cropped_images


def get_bin_number(points, nb_bins_per_dim, image_width, image_height):
    # get upper left point
    x_ul = np.max((points[0][0], points[1][0]))
    y_ul = np.max((points[0][1], points[1][1]))

    # get lower right point
    x_br = np.min((points[0][0], points[1][0]))
    y_br = np.min((points[0][1], points[1][1]))

    nb_pixels_per_bin_x = image_width / nb_bins_per_dim
    nb_pixels_per_bin_y = image_height / nb_bins_per_dim

    bin_x_ul = np.floor(x_ul / nb_pixels_per_bin_x)
    bin_y_ul = np.floor(y_ul / nb_pixels_per_bin_y)

    bin_ul = bin_x_ul + bin_y_ul * nb_bins_per_dim

    bin_x_br = np.floor(x_br / nb_pixels_per_bin_x)
    bin_y_br = np.floor(y_br / nb_pixels_per_bin_y)

    bin_br = bin_x_br + bin_y_br * nb_bins_per_dim

    return bin_ul, bin_br


def define_network(image_width, image_height, nb_bins):
    print('Defining model...')
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(image_height, image_width, 3), activation='relu', border_mode='same',
                            W_regularizer=l2(0.0005)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(nb_bins, W_regularizer=l2(0.02)))
    model.add(Activation('softmax'))

    optim = SGD(lr=0.0005, momentum=0.9, decay=0.0045, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model compiled.')
    return model


def train_network(network, images, labels, nb_bins):
    categorical = np_utils.to_categorical(labels, nb_bins)
    data_gen = image.ImageDataGenerator(rotation_range=10,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        featurewise_center=True,
                                        featurewise_std_normalization=True)
    data_gen.fit(images)

    # fits the model on batches with real-time data augmentation:
    network.fit_generator(data_gen.flow(images, categorical, batch_size=32),
                          samples_per_epoch=len(images), nb_epoch=100)

    network.save(filepath='saved_model', overwrite=True)


fish_type = 'alb'
points = preprocessing.load_json(fish_type)
cropped_images_of_fish = create_cropped_images_of_fish(points)
no_fish_pictures = subsample_from_no_fish_pictures()
category = np.concatenate((np.ones(len(cropped_images_of_fish)), np.zeros(len(no_fish_pictures))))
all_pictures = np.concatenate(
    (np.asarray(cropped_images_of_fish, dtype='float64'), np.asarray(no_fish_pictures, dtype='float64')))
nn = define_network(256, 256, 2)

train_network(nn, all_pictures, category, 2)

# first get json points.
# points = preprocessing.load_json()
# rescaled_image_size = 256
# nb_bins_per_dim = 16
# nb_bins = np.square(nb_bins_per_dim)
# image_names = points.keys()
# root_folder = r"C:\Users\Mark_\OneDrive\Documents\fish\train\ALB"
# scaled_img_root_folder = r"C:\Users\Mark_\OneDrive\Documents\fish\train\ScaledALB"
# scaled_images = []
#
# scaled_image_height = 128
# scaled_image_width = 256
#
# scaled_points_list = []
# upper_left_bin = []
# print('Processing images...')
# for img_name in image_names:
#    if len(points[img_name]) == 2:
#        full_path = os.path.join(root_folder, img_name)
#        scaled_image, scaled_points = scale_image(full_path, points[img_name])
#        ul, br = get_bin_number(scaled_points, nb_bins_per_dim, 256, 128)
#        upper_left_bin.append(ul)
#        scaled_images.append(np.asarray(scaled_image))
#        scaled_points_list.append(scaled_points)
#
#        # scaled_image_full_path = os.path.join(scaled_img_root_folder, img_name)
#        # scaled_image.save(scaled_image_full_path)
# print('Images processed.')
#
# neural_network = define_network(scaled_image_width, scaled_image_height, nb_bins)
# train_network(neural_network, scaled_images, upper_left_bin, nb_bins)

# for point in points
# get image for points with >= 2 points in json
# scale image
# scale points with image
# round points to nearest 'bin'
# pass to net to train!

# things to think about:
# how to deal with multiple fish. Worth excluding at first? Probably too many!
# see how this sloth thing works
# do we want to rotate fish?
