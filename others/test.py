import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


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

    return noisy_image

# file = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated/Dataset_Gauss_Simulated_Unwrapped/test/beam_nf/idx-17001__img-17182.png'
# file_noise = '/Users/jayden/Downloads/unwrapped_simulated_noise/Dataset_Gauss_Simulated_Unwrapped/test/beam_nf/idx-17001__img-17182.png'
file = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated/Dataset_Gauss_Simulated_Unwrapped/val/beam_ff/idx-14001__img-179.png'
file_noise = '/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated_noise/Dataset_Gauss_Simulated_Unwrapped/val/beam_ff/idx-14001__img-179.png'

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img_noise1 = cv2.imread(file_noise, cv2.IMREAD_GRAYSCALE)

img_noise = add_gaussian_noise(img, 0.16)

img_noise2 = np.clip(img_noise, 0, 255)  # 限制像素值范围
img_noise2_1 = img_noise2.astype(np.uint8)
cv2.imwrite('/Users/jayden/Desktop/test1.png', img_noise2_1)
img_noise2_2 = cv2.imread('/Users/jayden/Desktop/test1.png', cv2.IMREAD_GRAYSCALE)

img_noise3_1 = img_noise
cv2.imwrite('/Users/jayden/Desktop/test2.png', img_noise)
img_noise3_2 = cv2.imread('/Users/jayden/Desktop/test2.png', cv2.IMREAD_GRAYSCALE)

# 计算均方误差（MSE）
mse1 = np.sum((img - img_noise1) ** 2) / float(img.shape[0] * img.shape[1])
mse2_1 = np.sum((img - img_noise2_1) ** 2) / float(img.shape[0] * img.shape[1])
mse2_2 = np.sum((img - img_noise2_2) ** 2) / float(img.shape[0] * img.shape[1])
mse3_1 = np.sum((img - img_noise3_1) ** 2) / float(img.shape[0] * img.shape[1])
mse3_2 = np.sum((img - img_noise3_2) ** 2) / float(img.shape[0] * img.shape[1])
print(f'MSE1: {mse1}')
print(f'MSE2_1: {mse2_1}')
print(f'MSE2_2: {mse2_2}')
print(f'MSE3_1: {mse3_1}')
print(f'MSE3_2: {mse3_2}')

# 计算结构相似度指数（SSIM）
ssim1 = ssim(img, img_noise1)
ssim2_2 = ssim(img, img_noise2_2)
ssim3_2 = ssim(img, img_noise3_2)
print(f'SSIM1: {ssim1}')
print(f'SSIM2: {ssim2_2}')
print(f'SSIM3: {ssim3_2}')
