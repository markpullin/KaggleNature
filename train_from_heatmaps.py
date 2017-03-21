import os
from keras.models import Sequential
from keras.layers import Convolution3D, BatchNormalization, MaxPooling3D, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
import h5py
from keras.utils import np_utils
import test_script


def make_network(n_classes):
    model = Sequential()
    stride = 2
    model.add(Convolution3D(16, kernel_size=(3, 3, 3), input_shape=(32, 25, 3, 8), activation='relu', padding='same'))
    model.add(MaxPooling3D(strides=(stride, stride, 1)))
    model.add(Convolution3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(strides=(stride, stride, 1)))
    model.add(Convolution3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    optim = SGD(lr=0.0001, momentum=0.9, decay=0.00001, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, file_list, category_list):
    train_percentage = 0.5
    rand_idx = np.random.permutation(range(0, len(file_list)))
    train_idx = rand_idx[:int(len(file_list) * train_percentage)]
    val_idx = rand_idx[int(len(file_list) * train_percentage):]
    nb_epochs = 20
    batch_size = 32
    train_files = [file_list[i] for i in train_idx]
    val_files = [file_list[i] for i in val_idx]
    train_y_list = [category_list[i] for i in train_idx]
    val_y_list = [category_list[i] for i in val_idx]
    train_y = np.asarray(train_y_list)
    val_y = np.asarray(val_y_list)
    img_size = (32, 25, 3, 8)

    train_x = np.zeros((len(train_files), img_size[0], img_size[1], img_size[2], img_size[3]))
    for idx, file in enumerate(train_files):
        loaded_file = h5py.File(file, "r")
        temp = loaded_file.get('heatmap')
        train_x[idx, :, :, :, :] = np.transpose(np.asarray(temp), axes=[0, 1, 3, 2])

    val_x = np.zeros((len(val_files), img_size[0], img_size[1], img_size[2], img_size[3]))
    for idx, file in enumerate(val_files):
        loaded_file = h5py.File(file, "r")
        temp = loaded_file.get('heatmap')
        val_x[idx, :, :, :, :] = np.transpose(np.asarray(temp), axes=[0, 1, 3, 2])

    cat_train_y = np_utils.to_categorical(train_y, 8)
    cat_val_y = np_utils.to_categorical(val_y, 8)
    filepath = "heatmap-weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    model.fit(train_x, cat_train_y, epochs=nb_epochs, batch_size=batch_size,
              callbacks=[checkpoint], validation_data=(val_x, cat_val_y))


def train():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    heatmaps_dir = os.path.join(this_dir, "heatmaps")
    classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft', 'other', 'nof']
    folder_list = os.listdir(heatmaps_dir)
    category_list = []
    file_list = []
    for folder in folder_list:
        heatmap_full_folder = os.path.join(heatmaps_dir, folder)
        this_files = [os.path.join(heatmap_full_folder, file) for file in os.listdir(heatmap_full_folder)]
        file_list += this_files
        category = classes.index(folder.lower())
        category_list += [category] * len(this_files)
        print(len(this_files), 'files found')
        assert len(file_list) == len(category_list)

    model = make_network(len(classes))
    train_model(model, file_list, category_list)


def predict():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft', 'other', 'nof']
    model = make_network(8)
    folder = os.path.join(this_dir, 'trainheatmaps')
    files = os.listdir(folder)
    test_img = np.zeros((len(files), 32, 25, 3, 8))
    for idx, file in enumerate(files):
        loaded_file = h5py.File(file, "r")
        temp = loaded_file.get('heatmap')
        test_img[idx, :, :, :, :] = np.transpose(np.asarray(temp), axes=[0, 1, 3, 2])

    weights_path = os.path.join(this_dir, 'heatmap-weights-13.hdf5')
    model.load_weights(weights_path)
    probs = model.predict(test_img, batch_size=32)
    test_script.make_submission(probs, files, classes)

if __name__ == "__main__":
    predict()
