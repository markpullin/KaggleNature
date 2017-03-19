import PIL.Image as Image
import PIL.ImageDraw as Draw
import PIL.ImageFont as Fonts
import numpy as np
import matplotlib.pyplot as plt
import training_script
import preprocessing as pp


def test(model, image_file_names, scales_per_octave=3, stride=5):
    input_size = model.input_size

    patch_size = (299, 299)
    for image_name in image_file_names:
        test_at_constant_scale(model, image_name, patch_size, stride=5)

def normalise_image(x):
    x -= np.mean(x,axis=(0,1))[np.newaxis,np.newaxis,:]
    x /= 255
    return x

def test_at_constant_scale(model, image_name, patch_size, stride=5):
    image  = Image.open(image_name)
    font   = Fonts.truetype("arial.ttf", 20)
    drw    = Draw.Draw(image) 
    
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
            
            #plt.figure()
            #plt.imshow(array)
            #plt.show()
            #return
        
            array = normalise_image(array)
            array = array[np.newaxis,:,:,:]            
            predictions[i, j, :] = model.predict(array,batch_size=1)[np.newaxis, np.newaxis, :]            
            print(predictions[i, j, :])            
            drw.text((x_start,y_start),str(np.argmax(predictions[i, j, :])),(255,0,0),font)
            #drw.text((x_start+0.5*patch_size[0],y_start+0.5*patch_size[1]),str(np.argmax(predictions[i, j, :])),(255,0,0),font)
            #drw.text((x_start+patch_size[0],y_start+patch_size[1]),str(np.argmax(predictions[i, j, :])),(255,0,0)font)
            drw.rectangle([x_start,y_start,x_start+patch_size[0],y_start+patch_size[1]],outline='white')
    image.show()
    return predictions
            
            
def run():
    model = training_script.get_pre_trained_model(7)
    model.load_weights(r"C:\Users\Fifth\KaggleNature\new_weights-improvement-16.hdf5")
        
    predictions = test_at_constant_scale(model,r"C:\Users\Fifth\KaggleNature\train\SHARK\img_00033.jpg",[100,100],stride=25)
    
    predicted_class = np.argmax(predictions,axis=2)
    
    plt.figure()
    plt.imshow(predicted_class)
    plt.show()
    