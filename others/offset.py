# adding gaussian noise to the image
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import shift


def shift_image(image1, image2, shift_range=3):
    shift_x = np.random.randint(-shift_range, shift_range + 1)
    shift_y = np.random.randint(-shift_range, shift_range + 1)

    shifted_image1 = shift(image1, (shift_y, shift_x), mode='wrap')
    shifted_image2 = shift(image2, (shift_y, shift_x), mode='wrap')

    return shifted_image1, shifted_image2


def process_image(file_path, output_path):
    img1 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(file_path.replace('beam_nf', 'beam_ff'), cv2.IMREAD_GRAYSCALE)
    shifted_image1, shifted_image2 = shift_image(img1, img2)
    cv2.imwrite(output_path, shifted_image1)
    cv2.imwrite(output_path.replace('beam_nf', 'beam_ff'), shifted_image2)


# Folder with raw data
input_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated_noise'
output_folder = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated_shift'
different = ['unwrapped_simulated_noise', 'unwrapped_simulated_shift']

# Make directories
for dir1 in ['Dataset_Gauss_Simulated_Unwrapped', 'Dataset_Vortex_Simulated_Unwrapped']:
    for dir2 in ['train', 'val', 'test']:
        for dir3 in ['beam_ff', 'beam_nf']:
            os.makedirs(f"{output_folder}/{dir1}/{dir2}/{dir3}", exist_ok=True)

# Add origin images path and noisy images path
input_paths = []
output_paths = []
for subset, _, _ in os.walk(input_folder):
    if 'beam_nf' in subset and 'train' not in subset:
        for filename in os.listdir(subset):
            input_paths.append(os.path.join(subset, filename))
            output_paths.append(os.path.join(subset.replace(different[0], different[1]), filename))

# Create a ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor() as executor:
    for input_path, output_path in zip(input_paths, output_paths):
        executor.submit(process_image, input_path, output_path)

print("All images have been processed and saved.")
