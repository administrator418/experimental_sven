# adding gaussian noise to the image
import os
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
    noisy_image = np.clip(noisy_image, 0, 255)  # 限制像素值范围
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


def process_image(file_path, output_path, noise_sigma):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    noise_img = add_gaussian_noise(img, noise_sigma)
    cv2.imwrite(output_path, noise_img)


# Folder with raw data
input_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated/Dataset_Vortex_Simulated_Unwrapped/val'
output_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated_noise_0.16/Dataset_Vortex_Simulated_Unwrapped/val'
different = ['unwrapped_simulated', 'unwrapped_simulated_noise_0.16']

# Define the noise sigma
noise_sigma = 0.16  # 0.14, 0.16, 0.18

# Make directories
for dir1 in ['Dataset_Gauss_Simulated_Unwrapped', 'Dataset_Vortex_Simulated_Unwrapped']:
    for dir2 in ['train', 'val', 'test']:
        for dir3 in ['beam_ff', 'beam_nf']:
            os.makedirs(f"{'/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated_noise_0.16'}/{dir1}/{dir2}/{dir3}", exist_ok=True)

# Add origin images path and noisy images path
input_files = []
output_files = []
for subset, _, _ in os.walk(input_folder):
    if 'beam_ff' in subset or 'beam_nf' in subset:
        for filename in os.listdir(subset):
            input_files.append(os.path.join(subset, filename))
            output_files.append(os.path.join(subset.replace(different[0], different[1]), filename))

# Create a ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor() as executor:
    for input_file, output_file in zip(input_files, output_files):
        executor.submit(process_image, input_file, output_file, noise_sigma)

print("All images have been processed and saved.")