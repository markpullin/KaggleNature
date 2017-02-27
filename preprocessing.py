import os

from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np

import urllib, json

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
    left_boundary = np.round(rect_centre[0] - rect_width / 2)
    right_boundary = np.round(rect_centre[0] + rect_width / 2)
    top_boundary = np.round(rect_centre[1] - rect_height / 2)
    bottom_boundary = np.round(rect_centre[1] + rect_height / 2)
    width, height = image.size

    assert left_boundary >= 0
    assert right_boundary < width
    assert top_boundary >= 0
    assert bottom_boundary < height

    crop_area = (left_boundary, top_boundary, right_boundary, bottom_boundary)

    return image.crop(crop_area)


def draw_rectangle_on_image(image, rect_centre, rect_width, rect_height):
    ax = plt.axes()
    ax.imshow(image)
    top_left_corner = [None, None]
    top_left_corner[0] = rect_centre[0] - rect_width/2
    top_left_corner[1] = rect_centre[1] - rect_height/ 2
    rect = patches.Rectangle(top_left_corner, rect_width, rect_height, facecolor='None', edgecolor='white')
    ax.add_patch(rect)
    plt.show()


def load_json():
    json_path = r'C:\Users\Mark_\KaggleNature\kaggleNatureConservancy/alb_labels.json'
    json_file = open(json_path)
    data = json.load(json_file)
    d = dict()
    for cell in data:
        name = cell["filename"]
        points = cell["annotations"]

        points_tuple = []
        for point in points:
            points_tuple.append((point["x"], point["y"]))
        d[name] = points_tuple

    return d
