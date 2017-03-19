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
import test_script


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
    optim = SGD(lr=0.0001, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def subsample_from_no_fish_pictures(img_size):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    folder = os.path.join(dir_path, "train", "NoF")
    save_dir = os.path.join(dir_path, 'traincropped', "NoF")
    val_dir = os.path.join(dir_path, 'valcropped', "NoF")
    if os.path.isdir(save_dir):
        return

    os.mkdir(save_dir)
    os.mkdir(val_dir)
    files = os.listdir(folder)
    np.random.shuffle(files)
    val_idx = len(files) * 0.9
    images = []
    for idx, file in enumerate(files):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            img = Image.open(path)
            for i in range(0, 4):
                x_lower = int(np.ceil(np.random.rand(1) * (img.size[0] - img_size)))
                y_lower = int(np.ceil(np.random.rand(1) * (img.size[1] - img_size)))
                cropped = img.crop((x_lower, y_lower, x_lower + img_size, y_lower + img_size))
                # img = test_script.normalise_image(np.asarray(cropped).astype('float32'))
                # cropped = Image.fromarray(img.astype('uint8'))
                if idx < val_idx:
                    save_path = os.path.join(save_dir, str(i) + file)
                    cropped.save(save_path)
                else:
                    save_path = os.path.join(val_dir, str(i) + file)
                    cropped.save(save_path)

            for i in range(0, 4):
                rand_img_size = img_size * np.random.uniform(0.5, 2)
                x_lower = int(np.ceil(np.random.rand(1) * (img.size[0] - rand_img_size)))
                y_lower = int(np.ceil(np.random.rand(1) * (img.size[1] - rand_img_size)))
                cropped = img.crop((x_lower, y_lower, x_lower + rand_img_size, y_lower + rand_img_size))
                if idx < val_idx:
                    save_path = os.path.join(save_dir, 'zoom' + str(i) + file)
                    cropped.save(save_path)
                else:
                    save_path = os.path.join(val_dir, 'zoom' + str(i) + file)
                    cropped.save(save_path)
    return None


def create_cropped_images_of_fish(points, fish_type, img_size):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_folder = os.path.join(dir_path, os.path.join("train", fish_type.upper()))
    file_names = points.keys()
    if not os.path.isdir(os.path.join(dir_path, 'traincropped')):
        os.mkdir(os.path.join(dir_path, 'traincropped'))
    if not os.path.isdir(os.path.join(dir_path, 'valcropped')):
        os.mkdir(os.path.join(dir_path, 'valcropped'))
    save_dir = os.path.join(dir_path, 'traincropped', fish_type)
    val_dir = os.path.join(dir_path, 'valcropped', fish_type)
    if os.path.isdir(save_dir):
        return

    val_idx = len(file_names) * 0.9
    os.mkdir(save_dir)
    os.mkdir(val_dir)
    cropped_images = []
    for idx, file_name in enumerate(file_names):
        this_points = points[file_name]
        path = os.path.join(root_folder, file_name)
        if os.path.isfile(path) and (len(this_points) == 2):
            img = Image.open(path)
            x_br = np.min((this_points[0][0], this_points[1][0])) - 50
            y_br = np.min((this_points[0][1], this_points[1][1])) - 50
            x_tl = np.max((this_points[0][0], this_points[1][0])) + 50
            y_tl = np.max((this_points[0][1], this_points[1][1])) + 50
            x_range = x_tl - x_br
            y_range = y_tl - y_br
            x_br_range = np.linspace(x_br - x_range/2, x_br + x_range/2)
            y_br_range = np.linspace(y_br - y_range / 2, y_br + y_range / 2)
            for x_br_this in x_br_range:
                for y_br_this in x_br_range:
                    x_br_corrected = np.max((x_br_this, 0))
                    y_br_corrected = np.max((y_br_this, 0))
                    x_tl = np.min((x_br_corrected + x_range, img.size[0]))
                    y_tl = np.min((y_br_corrected + y_range, img.size[1]))

                    if x_range > y_range:
                        y_tl += (x_range - y_range) / 2
                        y_br -= (x_range - y_range) / 2
                    else:
                        x_tl += (y_range - x_range) / 2
                        x_br -= (y_range - x_range) / 2

                    cropped_img = img.crop((x_br, y_br, x_tl, y_tl))
                    cropped_img = cropped_img.resize((img_size, img_size))
                    # img_array = np.asarray(cropped_img)
                    # img_array = img_array.astype('float32')
                    # new_img_array = test_script.normalise_image(img_array)
                    # cropped_img = Image.fromarray(new_img_array.astype('uint8'))
                    if idx < val_idx:
                        save_path = os.path.join(save_dir, file_name)
                        cropped_img.save(save_path)
                    else:
                        save_path = os.path.join(val_dir, file_name)
                        cropped_img.save(save_path)


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

    optim = SGD(lr=0.0001, momentum=0.9, decay=0.00001, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model compiled.')
    return model


def train_network(network, train_folder, val_folder):
    filepath = "new_weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

    data_gen = image.ImageDataGenerator(rotation_range=10,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        )

    # fits the model on batches with real-time data augmentation:
    classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft', 'other', 'NoF']
    network.fit_generator(data_gen.flow_from_directory(train_folder,
                                                       classes=classes,
                                                       batch_size=32,
                                                       target_size=(299, 299)),
                          samples_per_epoch=6500, nb_epoch=40, callbacks=[checkpoint],
                          validation_data=data_gen.flow_from_directory(val_folder, target_size=(299, 299), classes=classes),
                          validation_steps=8)

    network.save(filepath='saved_model', overwrite=True)


def split_data_into_train_and_validation(images, category):
    X_train, X_test, y_train, y_test = train_test_split(images, category, train_size=0.8)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    fish_types = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft','other']
    cat = np.zeros(0)
    img_size = 299
    # count pictures

    dir_path = os.path.dirname(os.path.realpath(__file__))
    n_pictures = 0
    for fish_type in fish_types:
        root_folder = os.path.join(dir_path, os.path.join("train", fish_type.upper()))
        n_pictures += len(os.listdir(root_folder))

    for idx, fish_type in enumerate(fish_types):
        print('Loading json for ', fish_type)
        points = preprocessing.load_json(fish_type)
        print('json loaded. Getting fish from pictures...')
        create_cropped_images_of_fish(points, fish_type, img_size)

        print('Fish pictures obtained')
        cropped_images_of_fish = None
        points = None

    # get pictures of no fish separately - no json needed to locate fish
    subsample_from_no_fish_pictures(img_size)

    # nn = define_network(img_size, img_size, len(fish_types) + 1)
    nn = get_pre_trained_model(len(fish_types) + 1)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    train_folder = os.path.join(dir_path, 'traincropped')
    val_folder = os.path.join(dir_path, 'valcropped')
    train_network(nn, train_folder, val_folder)

    # class_totals = np.zeros(len(fish_types) + 1)
    # for iClass, _ in enumerate(class_totals):
    #     class_totals[iClass] = np.sum(test_labels is iClass)
    #
    # class_probabilities = nn.predict(test_images, batch_size=32)
    # predicted_classes = np.argmax(class_probabilities)
    # incorrect_predictions = np.not_equal(predicted_classes, test_labels)
    # wrong_classes_actual = test_labels[incorrect_predictions]
    #
    # plt.figure()
    # plt.hist(wrong_classes_actual)
    # plt.show()

    # wrong_classes_predicted = predicted_classes[incorrect_predictions]

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
