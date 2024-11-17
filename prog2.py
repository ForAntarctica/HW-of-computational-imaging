from PIL import Image
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
from ImageProcessor import ImageProcessor
import os

# 读取50张clean jpg图片
image_path = './clean_photos'
ImageProcessor = ImageProcessor(image_path, 50)
ImageProcessor.read_images()

# 随机的噪声污染这些图片
# 现在污染噪声的类型分为三种：椒盐噪声、高斯噪声、泊松噪声；10张图片只添加椒盐噪声，10张图片只添加高斯噪声，10张图片只添加泊松噪声，20张图片添加椒盐噪声、高斯噪声和泊松噪声
for i in range(10):
    salt_prob = random.random() * 0.01
    pepper_prob = random.random() * 0.01
    ImageProcessor.add_salt_pepper_noise(i, salt_prob, pepper_prob)


for i in range(10, 20):
    mean = 0
    sigma = random.uniform(0,10)
    ImageProcessor.add_gaussian_noise(i,mean,sigma)


for i in range(20, 30):
    ImageProcessor.add_poisson_noise(i)


for i in range(30, 50):
    salt_prob = random.random() * 0.01
    pepper_prob = random.random() * 0.01 
    mean = 0
    sigma = random.uniform(0,10)
    ImageProcessor.add_all_noises(i,salt_prob,pepper_prob,mean,sigma)


# ADMM去噪
AMDD_denoising_result = []
lam = 80
t1 = 0.1
t2 = 0.1
i = 16
AMDD_denoising_result.append(ImageProcessor.ADMM_denoising(i, lam, t1, t2, alpha_k=0.01))
plt.imshow(AMDD_denoising_result[0])
plt.title(f'PGD Denoising with λ={lam}')
plt.axis('off')  # 隐藏坐标轴
plt.show()
print('ADMM去噪完成')    