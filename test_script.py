import PIL.Image as Image
import numpy as np


def test(model, image_file_names, scales_per_octave=3, stride=5):
    input_size = model.input_size

    patch_size = (299, 299)
    for image_name in image_file_names:
        test_at_constant_scale(model, image_name, patch_size, stride=5)


def test_at_constant_scale(model, image_name, patch_size, stride=5):
    image = Image.open(image_name)
    size = image.size
    n_images_x = np.floor((size[0] - patch_size[0]) / stride).astype(int)
    n_images_y = np.floor((size[1] - patch_size[1]) / stride).astype(int)

    predictions = np.zeros((n_images_x, n_images_y, model.output_shape[1])
    
    for i in range(n_images_x):
        x_start = 0 + i * stride
        for j in range(n_images_y):
            y_start = 0 + j * stride
            cropped = image.crop((x_start, y_start, x_start + patch_size[0], y_start + patch_size[0]))
            resized = cropped.resize(patch_size)
            array = np.asarray(resized, dtype='float32')
            array -= np.mean(array, axis=3, keepdims=True)
            array /= np.std(array, axis=3, keepdims=True)
            predictions[i, j, :] = model.predict(array)[np.newaxis, np.newaxis, :]
            print('stop here')