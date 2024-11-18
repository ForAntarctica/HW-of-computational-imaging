import numpy as np
import random
import cv2 as cv
from skimage import color, data, restoration
import matplotlib.pyplot as plt
from Sparse_SIM import Sparse_SIM

# 初始化类
image_path = "E:\zhao_qy\zhaoqy_computation_imaging_hw\HW-of-computational-imaging\Microscopy_images"
image_name = "0000000.png"
Sparse_SIM = Sparse_SIM(image_path, image_name)

# 读取模糊图像
image_blur = Sparse_SIM.read_blur_image()

# 稀疏性解卷积
# 其中iteration_num为迭代次数，fidelity为保真项权重，sparse为稀疏性项权重，t为步长，ep为停止条件
x,g = Sparse_SIM.sparse_hessian_deconv(iteration_num = 1000, fidelity = 150, sparsity = 10, t = 0.01, ep = 1e-5, mu = 1)

# 显示结果
plt.figure()
plt.imshow(x, cmap='gray')
plt.title("Deconvolved Image")
plt.show()

plt.figure()
plt.imshow(g, cmap='gray')
plt.title("g Image")
plt.show()