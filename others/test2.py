import numpy as np
import cv2
from scipy.ndimage import shift

def shift_image(image1, image2, shift_range=3):
    # 随机生成x和y轴的位移量
    shift_x = np.random.randint(-shift_range, shift_range + 1)
    shift_y = np.random.randint(-shift_range, shift_range + 1)

    # 使用scipy的shift函数进行位移
    shifted_image1 = shift(image1, (shift_y, shift_x), mode='wrap')
    shifted_image2 = shift(image2, (shift_y, shift_x), mode='wrap')

    return shifted_image1, shifted_image2

# 使用例子
image1 = cv2.imread('/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated/Dataset_Gauss_Simulated_Unwrapped/train/beam_ff/idx-1__img-10651.png', cv2.IMREAD_GRAYSCALE)  # 读入第一张图片
image2 = cv2.imread('/Users/jayden/Documents/Jikai_Wang/unwrapped_simulated/Dataset_Gauss_Simulated_Unwrapped/train/beam_nf/idx-1__img-10651.png', cv2.IMREAD_GRAYSCALE)  # 读入第二张图片

shifted_image1, shifted_image2 = shift_image(image1, image2, shift_range=3)

cv2.imwrite('/Users/jayden/Desktop/shifted_image1.jpg', shifted_image1)
cv2.imwrite('/Users/jayden/Desktop/shifted_image2.jpg', shifted_image2)
