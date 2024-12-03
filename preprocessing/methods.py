from PIL import Image
import numpy as np
import os

def crop_around_center(image_path, save_path, crop_width=512, crop_height=512):
    image = Image.open(image_path)
    image_np = np.array(image)
    center_y, center_x = image_np.shape[0] // 2, image_np.shape[1] // 2

    start_x = max(center_x - crop_width // 2, 0)
    start_y = max(center_y - crop_height // 2, 0)
    
    end_x = min(start_x + crop_width, image_np.shape[1])
    end_y = min(start_y + crop_height, image_np.shape[0])

    cropped_image_np = image_np[start_y:end_y, start_x:end_x]
    
    cropped_image = Image.fromarray(cropped_image_np)
    
    cropped_image.save(save_path)
    
    return cropped_image

def normalize(img):
    
    normalized_img = img/255

    return normalized_img

def has_pixel_above_threshold(image_path, threshold = 50):
    img = Image.open(image_path)
    img_array = np.array(img)
    img_array = img_array/ np.max(img_array)
    above_threshold = np.any(img_array > threshold)
    return(above_threshold)

def oversaturation(image_path, threshold = 50):
    img = Image.open(image_path)
    img_array = np.array(img)
    above_threshold = np.any(img_array > threshold)
    return(above_threshold)


def crop_beam(img, center_of_mass,crop_size=256): # alt 128
    left = round(max(center_of_mass[1] - crop_size // 2, 0)) 
    top = round(max(center_of_mass[0] - crop_size // 2, 0))
    right = round(min(center_of_mass[1] + crop_size // 2, img.shape[0]))
    bottom = round(min(center_of_mass[0] + crop_size // 2, img.shape[1]))

    img = img[top:bottom, left:right]

    return img


def background_substraction(image_path, region_size=256):  # old 30
    img = Image.open(image_path)
    img_array = np.array(img)
    right_edge_region = img_array[:, -region_size:]
    mean_pixel = np.mean(right_edge_region)
    bgs_final_img_array = img_array - mean_pixel
    bgs_final_img = Image.fromarray(bgs_final_img_array)
    return(bgs_final_img)


def create_black_images(num_images, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    width, height = 128, 128

    for i in range(num_images):
        image = Image.new('RGB', (width, height), 'black')
        filename = f"black_image_{i+1:05d}.png"
        image.save(os.path.join(directory, filename))