# adding gaussian noise to the image
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = image_in.astype(np.float64)
    h, w = temp_image.shape[0], temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    """
    Debugging
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image


def process_image(file_path, output_path, noise_sigma):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    noise_img = add_gaussian_noise(img, noise_sigma)
    cv2.imwrite(output_path, noise_img)


# Folder with raw data
input_folder = "/Users/jayden/Desktop/vortex_0.16_0.16/val"
output_folder = "/Users/jayden/Desktop/vortex_0.16_0.16/val_noise"
different = ['val', 'val_noise']

# Define the noise sigma
""" Labor sigma = 0.14
    Test_1 sigma = 0.16
    Test_2 sigma = 0.18
    """
noise_sigma = 0.16

# Add origin images path and noisy images path
input_files ,output_files = [], []
for subset, _, _ in os.walk(input_folder):
    if 'beam_ff' in subset or 'beam_nf' in subset:
        for filename in os.listdir(subset):
            input_files.append(os.path.join(subset, filename))
            output_files.append(os.path.join(subset.replace(different[0], different[1]), filename))

# Make directories
for dir1 in ['beam_ff', 'beam_nf']:
    os.makedirs(f"{output_folder}/{dir1}", exist_ok=True)

# Create a ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor() as executor:
    for input_file, output_file in zip(input_files, output_files):
        executor.submit(process_image, input_file, output_file, noise_sigma)

print("All images have been processed and saved.")
