import test_script
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv3D

import training_script


def make_heat_maps(model, image_name, standard_image_size=(1280, 1000), stride=20, n_classes=8):
    scales = [140, 220, 300, 380, 460, 540]
    img = Image.open(image_name)
    img_size = img.size
    resize_ratio = img_size[0]/standard_image_size[0]
    img_resized = img.resize((standard_image_size[0], int(img_size[1]/resize_ratio)))
    img_array = np.zeros((standard_image_size[1], standard_image_size[0], 3), dtype='uint8')
    img_array[:img_resized.size[1], :, :] = np.asarray(img_resized)
    img = Image.fromarray(img_array)

    n_images_x = np.ceil(standard_image_size[0] / stride).astype(int)
    n_images_y = np.ceil(standard_image_size[1] / stride).astype(int)
    heat_maps = np.zeros((n_images_x, n_images_y,  n_classes, len(scales)))
    for idx, scale in enumerate(scales):
        raw_map = test_script.test_at_constant_scale(model, img, (scale, scale), stride)
        #reshaped = heat_maps[:, :, :, idx]
        idx_offset = int(scale/stride)
        #reshaped[idx_offset:n_images_x-idx_offset, idx_offset:n_images_y-idx_offset, :] \
        #    = np.reshape(raw_map, (n_images_x-idx_offset*2, n_images_y-idx_offset*2, n_classes))

        map_reshaped = np.reshape(raw_map, (n_images_x-idx_offset, n_images_y-idx_offset, n_classes))
        scale_over_stride = scale / stride
        counter = np.zeros(heat_maps.shape[:2])
        for i in range(0, map_reshaped.shape[0]):
            for j in range(0, map_reshaped.shape[1]):
                x_start = int(i)
                x_end = int(x_start + scale_over_stride)
                y_start = int(j)
                y_end = int(y_start + scale_over_stride)
                heat_maps[x_start:x_end, y_start:y_end, :, idx] += map_reshaped[int(i), int(j)]
                counter[x_start:x_end, y_start:y_end] += 1

        heat_maps[:, :, :, idx] /= counter[:, :, np.newaxis]
    return

def create_network():
    model = Sequential()
    #model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2)))
    return

if __name__ == "__main__":
    classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft', 'other', 'NoF']
    model = training_script.get_pre_trained_model(len(classes))
    model.load_weights(r"C:\Users\Fifth\KaggleNature\best_weights2.hdf5")
    make_heat_maps(model, r"C:\Users\Fifth\KaggleNature\train\ALB\img_00012.jpg")