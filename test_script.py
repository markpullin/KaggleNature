import PIL.Image as Image
import PIL.ImageDraw as Draw
import PIL.ImageFont as Fonts
import numpy as np
import matplotlib.pyplot as plt
import training_script
import preprocessing as pp
import csv
import pandas as pd
import os


def make_submission(probabilities, img_names, fish_names):
    file_name = 'sumbission'
    submission = pd.DataFrame(probabilities, columns=fish_names)
    submission.insert(0, 'image', img_names)
    submission.head()
    submission.to_csv(file_name)

def test(model, image_file_names, scales_per_octave=3, stride=5):
    input_size = model.input_size

    patch_size = (299, 299)
    for image_name in image_file_names:
        test_at_constant_scale(model, image_name, patch_size, stride=5)


def normalise_image(x):
    x -= np.mean(x, axis=(0, 1))[np.newaxis, np.newaxis, :]
    x /= 255
    return x


def test_at_constant_scale(model, image_name, patch_size, stride=5):
    image = Image.open(image_name)
    font = Fonts.truetype("arial.ttf", 20)
    drw = Draw.Draw(image)

    size = image.size
    n_images_x = np.ceil((size[0] - patch_size[0]) / stride).astype(int)
    n_images_y = np.ceil((size[1] - patch_size[1]) / stride).astype(int)
    predictions = np.zeros((n_images_x, n_images_y, model.output_shape[1]))

    for i in range(n_images_x):
        x_start = 0 + i * stride
        for j in range(n_images_y):
            y_start = 0 + j * stride
            cropped = image.crop((x_start, y_start, x_start + patch_size[0], y_start + patch_size[0]))

            resized = cropped.resize(model.input_shape[1:3])
            array = np.asarray(resized, dtype='float32')

            # plt.figure()
            # plt.imshow(array)
            # plt.show()
            # return

            #array = normalise_image(array)
            array = array[np.newaxis, :, :, :]
            predictions[i, j, :] = model.predict(array, batch_size=1)[np.newaxis, np.newaxis, :]
            if predictions[i, j, 0]  < 0.5:
                print(predictions[i, j, :])
            drw.text((x_start, y_start), str(np.argmax(predictions[i, j, :])), (255, 0, 0), font)
            # drw.text((x_start+0.5*patch_size[0],y_start+0.5*patch_size[1]),str(np.argmax(predictions[i, j, :])),(255,0,0),font)
            # drw.text((x_start+patch_size[0],y_start+patch_size[1]),str(np.argmax(predictions[i, j, :])),(255,0,0)font)
            drw.rectangle([x_start, y_start, x_start + patch_size[0], y_start + patch_size[1]], outline='white')
    image.show()
    return predictions

if __name__ == "__main__":
    classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft', 'other', 'NoF']
    model = training_script.get_pre_trained_model(len(classes))

    model.load_weights(r"C:\Users\Fifth\KaggleNature\best_weights.hdf5")
    test_folder = r"C:\Users\Fifth\KaggleNature\test"
    list_dir = os.listdir(test_folder)
    predicted_classes = np.zeros((len(list_dir), len(classes)))
    for idx, file in enumerate(list_dir):
        predictions = test_at_constant_scale(model, file, [600, 600], stride=20)
        new_predictions = np.reshape(predictions, (predictions.shape[0] * predictions.shape[1], len(classes)))
        crap_prediction_for_image = new_predictions[np.argmin(new_predictions, axis=0)[0], :]
        predicted_classes[idx, :] = crap_prediction_for_image[:, np.newaxis]
    make_submission(predicted_classes, list_dir, classes)

    # predicted_class = np.argmax(predictions, axis=2)
    # plt.figure()
    # plt.imshow(predicted_class)
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(predictions[:,:,2])
    # plt.colorbar()
    # plt.show()
