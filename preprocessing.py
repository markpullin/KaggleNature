import os

from PIL import Image


def load_all_images_from_folder(folder, n_max):
    list_of_files = os.listdir(folder)

    images = []
    for file_name in list_of_files:
        full_path = os.path.join(folder, file_name)
        im = Image.open(full_path)
        images.append(im)

    return images
