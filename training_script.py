import os

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Input, \
    GlobalAveragePooling2D, Dropout
from keras.models import Sequential, Model
from keras.applications import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import preprocessing


def get_pre_trained_model(nb_classes):
    img_width, img_height = 299, 299

    input = Input(shape=(img_width, img_height, 3))

    incep3 = InceptionV3(include_top=False)
    x = incep3(input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predict = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=predict)
    optim = SGD(lr=0.0005, momentum=0.9, decay=0.0045, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def subsample_from_no_fish_pictures(img_size):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    folder = os.path.join(dir_path, "train", "NoF")
    files = os.listdir(folder)
    images = []
    for file in files:
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            img = Image.open(path)
            x_lower = int(np.ceil(np.random.rand(1) * (img.size[0] - img_size)))
            y_lower = int(np.ceil(np.random.rand(1) * (img.size[1] - img_size)))
            cropped = img.crop((x_lower, y_lower, x_lower + img_size, y_lower + img_size))
            images.append(np.asarray(cropped))
    return images


def create_cropped_images_of_fish(points, fish_type, img_size):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_folder = os.path.join(dir_path, os.path.join("train", fish_type.upper()))
    file_names = points.keys()

    cropped_images = []
    for file_name in file_names:
        this_points = points[file_name]
        path = os.path.join(root_folder, file_name)
        if os.path.isfile(path) and (len(this_points) == 2):
            img = Image.open(path)
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
            cropped_img = cropped_img.resize((img_size, img_size))
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


def define_network(image_width, image_height, nb_classes):
    print('Defining model...')

    pool_size = (3, 3)
    pool_strides = (2, 2)
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(image_height, image_width, 3), activation='relu', border_mode='same',
                            W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
    model.add(Flatten())
    model.add(Dense(nb_classes, W_regularizer=l2(0.01)))
    model.add(Activation('softmax'))

    optim = SGD(lr=0.0005, momentum=0.9, decay=0.0045, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model compiled.')
    return model


def train_network(network, train_images, train_labels, test_images, test_labels, nb_classes):
    
    
    train_images -= np.mean(train_images, axis=(1,2), keepdims=True)
    test_images -= np.mean(test_images, axis=(1,2), keepdims=True)
    train_images /= np.std(train_images, axis=(1,2), keepdims=True)
    test_images /= np.std(test_images, axis=(1, 2), keepdims=True)


    filepath = "new_weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

    train_labels_cat = np_utils.to_categorical(train_labels, nb_classes)
    test_labels_cat = np_utils.to_categorical(test_labels, nb_classes)
    data_gen = image.ImageDataGenerator(rotation_range=10,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2)


    data_gen.fit(train_images[::10])

    # fits the model on batches with real-time data augmentation:
    network.fit_generator(data_gen.flow(train_images, train_labels_cat, batch_size=32),
                          samples_per_epoch=len(train_images), nb_epoch=20, callbacks=[checkpoint],
                          validation_data=data_gen.flow(test_images, test_labels_cat),validation_steps = 8)

    network.save(filepath='saved_model', overwrite=True)


def split_data_into_train_and_validation(images, category):
    X_train, X_test, y_train, y_test = train_test_split(images, category, train_size=0.8)
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    fish_types = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft']
    cat = np.zeros(0)
    img_size = 299
    # count pictures

    dir_path = os.path.dirname(os.path.realpath(__file__))
    n_pictures = 0
    for fish_type in fish_types:
        root_folder = os.path.join(dir_path, os.path.join("train", fish_type.upper()))
        n_pictures += len(os.listdir(root_folder))

    n_pictures += len(os.listdir(os.path.join(dir_path, os.path.join("train", "NoF"))))
    print(n_pictures, ' pictures found')
    all_pictures = np.zeros((n_pictures, img_size, img_size, 3))
    i_pic = 0
    for idx, fish_type in enumerate(fish_types):
        print('Loading json for ', fish_type)
        points = preprocessing.load_json(fish_type)
        print('json loaded. Getting fish from pictures...')
        cropped_images_of_fish = create_cropped_images_of_fish(points, fish_type, img_size)
        cat = np.concatenate((cat, np.ones(len(cropped_images_of_fish)) * idx))
        n_pic = len(cropped_images_of_fish)
        all_pictures[i_pic:(i_pic+n_pic), :, :, :] = np.asarray(cropped_images_of_fish).astype('float32')
        i_pic += n_pic
        print('Fish pictures obtained')
        cropped_images_of_fish = None
        points = None

    # get pictures of no fish separately - no json needed to locate fish
    no_fish_pictures = subsample_from_no_fish_pictures(img_size)
    cat = np.concatenate((cat, np.ones(len(no_fish_pictures)) * len(fish_types)))
    n_pic = len(no_fish_pictures)
    all_pictures[i_pic:(i_pic+n_pic), :, :, :] = np.asarray(no_fish_pictures, dtype='float32')
    no_fish_pictures = None
    
    all_pictures = all_pictures[:i_pic+n_pic,:,:,:]

    print('Picture size is', all_pictures.shape)
    print('Label size is', cat.shape)
    train_images, train_labels, test_images, test_labels = split_data_into_train_and_validation(all_pictures, cat)
    all_pictures = None
    
    #nn = define_network(img_size, img_size, len(fish_types) + 1)
    nn = get_pre_trained_model(len(fish_types) + 1)
    train_network(nn, train_images, train_labels, test_images, test_labels, len(fish_types) + 1)

    class_totals = np.zeros(len(fish_types)+1)
    for iClass,_ in enumerate(class_totals):
        class_totals[iClass] = np.sum(test_labels is iClass)
    

    class_probabilities     = nn.predict(test_images,batch_size=32)
    predicted_classes       = np.argmax(class_probabilities)
    incorrect_predictions   = np.not_equal(predicted_classes, test_labels)
    wrong_classes_actual    = test_labels[incorrect_predictions]

    plt.figure()
    plt.hist(wrong_classes_actual)
    plt.show()

    #wrong_classes_predicted = predicted_classes[incorrect_predictions]

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
