from PIL import Image
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
from ImageProcessor import ImageProcessor
import os
import bm3d
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import color, data, restoration

image_path = "./noisy_photos"
image_output_path = "./bonus_output"
noisy_images = []
num_images = 200
for i in range(num_images):
    image = Image.open(f'{image_path}/{i}.jpg')
    noisy_images.append(image)

bm3d_filter_result = []
for i in range(200):
    image = noisy_images[i]
    image = np.array(image)

    # 每个通道先中值滤波
    image = cv.medianBlur(image, 5)
    # 再均值滤波
    image = cv.blur(image, (5, 5))

    # bm3d算法去噪
    sigma = restoration.estimate_sigma(image,channel_axis=-1)
    image = bm3d.bm3d_rgb(image, sigma)
    image = np.clip(image, 0, 255).astype(np.uint8)
    bm3d_filter_result.append(Image.fromarray(image))
    image = Image.fromarray(image)  
    print(f'{i}')
    # 在指定路径保存图片
    image.save(f'{image_output_path}/bm3d_{i}.jpg')
