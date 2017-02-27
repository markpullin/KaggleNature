import os

from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np


def load_all_images_from_folder(folder, n_max):
    list_of_files = os.listdir(folder)
    n = np.min((len(list_of_files), n_max))
    list_of_files = list_of_files[:n]

    images = []
    for file_name in list_of_files:
        full_path = os.path.join(folder, file_name)
        im = Image.open(full_path)
        images.append(im)

    return images


def take_rectangle_of_image(image, rect_centre, rect_width, rect_height):
    width, height   = image.size
    
    left_boundary   = np.round(rect_centre[0] - rect_width / 2)
    top_boundary    = np.round(rect_centre[1] - rect_height / 2)
    right_boundary  = np.round(rect_centre[0] + rect_width / 2)
    bottom_boundary = np.round(rect_centre[1] + rect_height / 2)

    left_boundary   = np.minimum(width - rect_width,   np.maximum(0,           left_boundary))
    top_boundary    = np.minimum(height - rect_height, np.maximum(0,           top_boundary))
    right_boundary  = np.minimum(width,                np.maximum(rect_width,  right_boundary))
    bottom_boundary = np.minimum(height,               np.maximum(rect_height, top_boundary))
   
    crop_area = (left_boundary, top_boundary, right_boundary, bottom_boundary)

    return image.crop(crop_area)


def draw_rectangle_on_image(image, rect_centre, rect_width, rect_height):
    ax = plt.axes()
    ax.imshow(image)
    width, height = image.size

    top_left_corner    = [None, None]
    top_left_corner[0] = np.minimum(width-rect_width,   np.maximum(0, rect_centre[0] - rect_width/2))
    top_left_corner[1] = np.minimum(height-rect_height, np.maximum(0, rect_centre[1] - rect_height/ 2))
   
    rect = patches.Rectangle(top_left_corner, rect_width, rect_height, facecolor='None', edgecolor='white')
    ax.add_patch(rect)
    plt.show()

