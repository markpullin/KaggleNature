import os

from PIL import Image, Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
import json


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
    width, height = image.size

    left_boundary = np.round(rect_centre[0] - rect_width / 2)
    top_boundary = np.round(rect_centre[1] - rect_height / 2)
    right_boundary = np.round(rect_centre[0] + rect_width / 2)
    bottom_boundary = np.round(rect_centre[1] + rect_height / 2)

    left_boundary = np.minimum(width - rect_width, np.maximum(0, left_boundary))
    top_boundary = np.minimum(height - rect_height, np.maximum(0, top_boundary))
    right_boundary = np.minimum(width, np.maximum(rect_width, right_boundary))
    bottom_boundary = np.minimum(height, np.maximum(rect_height, bottom_boundary))

    crop_area = (left_boundary, top_boundary, right_boundary, bottom_boundary)

    return image.crop(crop_area)


def draw_rectangle_on_image(image, rect_centre, rect_width, rect_height):
    ax = plt.axes()
    ax.imshow(image)
    width, height = image.size

    top_left_corner = [None, None]
    top_left_corner[0] = np.minimum(width - rect_width, np.maximum(0, rect_centre[0] - rect_width / 2))
    top_left_corner[1] = np.minimum(height - rect_height, np.maximum(0, rect_centre[1] - rect_height / 2))

    rect = patches.Rectangle(top_left_corner, rect_width, rect_height, facecolor='None', edgecolor='white')
    ax.add_patch(rect)
    plt.show()


def load_json(fish_type):
    this_path = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(this_path, fish_type + '_labels.json')
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


def plot_points_on_image(folder, n_images):
    points = load_json()

    list_of_files = os.listdir(folder)
    n = np.min((len(list_of_files), n_images))
    list_of_files = list_of_files[:n]

    for file_name in list_of_files:
        full_path = os.path.join(folder, file_name)
        im = Image.open(full_path)

        this_points = points[file_name]
        if len(this_points) is 2:
            rect_centre = ((this_points[0][0] + this_points[1][0]) / 2, (this_points[0][1] + this_points[1][1]) / 2)
            new_im = take_rectangle_of_image(im, rect_centre, 400, 400)
            plt.figure()
            plt.imshow(new_im)
            plt.show()

        for point in this_points:
            plt.scatter(point[0], point[1])

    return None


def scale_image(img, points):
    # scale points and image
    target_width = 256
    target_height = 128

    original_image = Image.open(img)
    original_size = original_image.size
    scaled_image = original_image.resize((target_width, target_height))
    scaling_factor = (target_width / original_size[0], target_height / original_size[1])

    scaled_points = []
    for point in points:
        # TODO: check index -> x/y mapping
        scaled_point = (point[0] * scaling_factor[0], point[1] * scaling_factor[1])
        scaled_points.append(scaled_point)

    return scaled_image, scaled_points
