# adding gaussian noise to the image
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


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


def process(file_path, output_path, noise_sigma):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    noise_img = add_gaussian_noise(img, noise_sigma)
    cv2.imwrite(output_path, noise_img)


# Folder with raw data
input_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated'
output_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated_noise0.2'
different = ['unwrapped_simulated', 'unwrapped_simulated_noise0.2']

# Define the noise sigma
noise_sigma = 0.20  # 0.14, 0.16, 0.18

# Make directories
for dir1 in ['Dataset_Gauss_Simulated_Unwrapped', 'Dataset_Vortex_Simulated_Unwrapped']:
    for dir2 in ['test']:
        for dir3 in ['beam_ff', 'beam_nf']:
            os.makedirs(f"{output_folder}/{dir1}/{dir2}/{dir3}", exist_ok=True)

# Add origin images path and noisy images path
input_paths, output_paths = [], []
for subset, _, _ in os.walk(input_folder):
    if 'beam_nf' in subset and 'test' in subset:
        for filename in os.listdir(subset):
            input_paths.append(os.path.join(subset, filename))
            output_paths.append(os.path.join(subset.replace(different[0], different[1]), filename))
            input_paths.append(os.path.join(subset.replace('beam_nf', 'beam_ff'), filename))
            output_paths.append(os.path.join(subset.replace('beam_nf', 'beam_ff').replace(different[0], different[1]), filename))

# Create a ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor() as executor:
    futures = []
    for input_path, output_path in zip(input_paths, output_paths):
        future = executor.submit(process, input_path, output_path, noise_sigma)
        futures.append(future)

    for future in tqdm(futures, desc="Processing", total=len(futures)):
        future.result()

print("All images have been processed and saved.")
