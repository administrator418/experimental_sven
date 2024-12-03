#adding gaussian noise to the image 
import cv2
import numpy as np
import cv2
import os
import glob
import skimage
import numpy as np
import matplotlib.pyplot as plt
import math

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
    """
    Debugging
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image

# Folder with raw data
input_folder = r"\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Datasets\Simulativ\Raw_Data_Gauusian_Intensity_Beam_Test"
output_folder = r"\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Datasets\Simulativ\Raw_Data_Gauusian_Intensity_Beam_Noise_Sigma028"

# Define the noise sigma
""" Labor sigma = 0.14
    Test_1 sigma = 0.16
    Test_2 sigma = 0.18
    """
noise_sigma = 0.16

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        file_path = os.path.join(input_folder, filename)
        img = cv2.imread(file_path)
        noise_img = add_gaussian_noise(img, noise_sigma=noise_sigma)
        output_file_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_file_path, noise_img)

print("All images have been processed and saved.")