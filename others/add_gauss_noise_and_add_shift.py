# adding gaussian noise to the image
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import shift


def process_image(img1, img2, noise_sigma, shift_range=3):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    h, w = img1.shape[0], img1.shape[1]
    noise1 = np.random.randn(h, w) * noise_sigma
    noise2 = np.random.randn(h, w) * noise_sigma

    processing_img1 = np.zeros(img1.shape, np.float64)
    processing_img2 = np.zeros(img2.shape, np.float64)

    shift_x = np.random.randint(-shift_range, shift_range + 1)
    shift_y = np.random.randint(-shift_range, shift_range + 1)

    processing_img1 = shift(processing_img1, (shift_y, shift_x), mode='wrap')
    processing_img2 = shift(processing_img2, (shift_y, shift_x), mode='wrap')

    if len(img1.shape) == 2:
        processing_img1 = img1 + noise1
        processing_img2 = img2 + noise2
    else:
        processing_img1[:, :, 0] = img1[:, :, 0] + noise1
        processing_img1[:, :, 1] = img1[:, :, 1] + noise1
        processing_img1[:, :, 2] = img1[:, :, 2] + noise1

        processing_img2[:, :, 0] = img2[:, :, 0] + noise2
        processing_img2[:, :, 1] = img2[:, :, 1] + noise2
        processing_img2[:, :, 2] = img2[:, :, 2] + noise2

    processed_img1 = np.clip(processing_img1, 0, 255).astype(np.uint8)
    processed_img2 = np.clip(processing_img2, 0, 255).astype(np.uint8)

    return processed_img1, processed_img2


def process(file_path, output_path, noise_sigma):
    img1 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(file_path.replace('beam_nf', 'beam_ff'), cv2.IMREAD_GRAYSCALE)
    processed_img1, processed_img2 = process_image(img1, img2, noise_sigma)
    cv2.imwrite(output_path, processed_img1)
    cv2.imwrite(output_path, processed_img2)


# Folder with raw data
input_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated'
output_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated_noise'
different = ['unwrapped_simulated', 'unwrapped_simulated_noise']

# Define the noise sigma
noise_sigma = 0.16  # 0.14, 0.16, 0.18

# Make directories
for dir1 in ['Dataset_Gauss_Simulated_Unwrapped', 'Dataset_Vortex_Simulated_Unwrapped']:
    for dir2 in ['val', 'test']:
        for dir3 in ['beam_ff', 'beam_nf']:
            os.makedirs(f"{output_folder}/{dir1}/{dir2}/{dir3}", exist_ok=True)

# Add origin images path and noisy images path
input_paths, output_paths = [], []
for subset, _, _ in os.walk(input_folder):
    if 'beam_nf' in subset and 'train' not in subset:
        for filename in os.listdir(subset):
            input_paths.append(os.path.join(subset, filename))
            output_paths.append(os.path.join(subset.replace(different[0], different[1]), filename))

# Create a ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor() as executor:
    for input_path, output_path in zip(input_paths, output_paths):
        executor.submit(process, input_path, output_path, noise_sigma)

print("All images have been processed and saved.")
