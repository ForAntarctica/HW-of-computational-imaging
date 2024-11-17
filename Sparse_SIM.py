import numpy as np
import random
import cv2 as cv
from skimage import color, data, restoration
import matplotlib.pyplot as plt

class Sparse_SIM:
    # 该算法是针对2D图像的Sparse算法
    def __init__(self, image_path, image_name):
        self.image_path = image_path
        self.image_name = image_name
        self.image_size = []
        self.blur_image = []
    def read_blur_image(self):
        blur_image = cv.imread(self.image_path + self.image_name)
        # 将RGB图像转换为灰度图
        blur_image = cv.cvtColor(blur_image, cv.COLOR_BGR2GRAY)
        # 将灰度图像转换为浮点数
        blur_image = blur_image.astype(np.float64) / 255   
        self.blur_image = blur_image.copy() 
        self.image_size = np.shape(self.blur_image)
        return blur_image
    # 一阶范数的软阈值函数
    def shrink(self,x,mu):
        s = np.abs(x)
        xs = np.sign(x) * np.maximum(s - 1 /mu,0)
        return xs
    # 2D图像海森矩阵的一阶范数的优化迭代步
    def iter_xx(self,g_k_old,bxx,mu):
        gxx = np.diff(np.diff(g_k_old,axis=1),axis=1)
        dxx = Sparse_SIM.shrink(gxx + bxx, mu)
        bxx = bxx + (gxx - dxx)
        Lxx = np.diff(np.diff(dxx - bxx,axis=1),axis=1)
        return Lxx, bxx
    def iter_xy(self,g_k_old,bxy,mu):
        gxy = np.diff(np.diff(g_k_old,axis=1),axis=0)
        dxy = Sparse_SIM.shrink(gxy + bxy, mu)
        bxy = bxy + (gxy - dxy)
        Lxy = np.diff(np.diff(dxy - bxy,axis=0),axis=1)
        return Lxy, bxy
    def iter_yx(self,g_k_old,byx,mu):
        gyx = np.diff(np.diff(g_k_old,axis=0),axis=1)
        dyx = Sparse_SIM.shrink(gyx + byx, mu)
        byx = byx + (gyx - dyx)
        Lyx = np.diff(np.diff(dyx - byx,axis=1),axis=0)
        return Lyx, byx    
    def iter_yy(self,g_k_old,byy,mu):
        gyy = np.diff(np.diff(g_k_old,axis=0),axis=0)
        dyy = Sparse_SIM.shrink(gyy + byy, mu)
        byy = byy + (gyy - dyy)
        Lyy = np.diff(np.diff(dyy - byy,axis=0),axis=0)
        return Lyy, byy
    # L1范数的优化迭代步
    def iter_L1(self,g_k_old,bL1,sparse,mu):
        dL1 = Sparse_SIM.shrink(g_k_old + bL1,mu)
        bL1 = bL1 + (g_k_old - dL1)
        LL1 = sparse * (dL1 - bL1)
        return LL1, bL1
    
    def sparse_hessian_deconv(self, iteration_num, fidelity, sparsity,t,ep,mu):
        ''' 
        该算法是先进行正则化的优化，优化的目标函数为：argmin_g = 
        { fidelity / 2 * ||f-g||_2^2 + ||gxx||_1 + 2*||gxy||_1 + ||gyy||_1 + sparsity * ||g||_1}
        '''
        # 采用PGD算法进行优化
        f = self.blur_image
        g_k = f.copy()
        g_k_old = f.copy()

        ## initialize b
        bxx = np.zeros(self.image_size,dtype='float64')
        byy = bxx
        bxy = bxx
        byx = bxx
        bL1 = bxx

        for i in range(iteration_num):
            # 梯度下降
            g_k_temp = g_k_old - t * fidelity * (g_k_old - f)

            # 考虑海森正则项的次梯度下降
            Lxx,bxx = self.iter_xx(g_k_old, bxx, mu)
            g_k_temp = g_k_temp + Lxx
            del Lxx
            Lxy,bxy = self.iter_xy(g_k_old, bxy, mu)
            g_k_temp = g_k_temp + Lxy
            del Lxy
            Lyx,byx = self.iter_yx(g_k_old, byx, mu)
            g_k_temp = g_k_temp + Lyx
            del Lyx
            Lyy,byy = self.iter_yy(g_k_old, byy, mu)
            g_k_temp = g_k_temp + Lyy
            del Lyy

            # 考虑稀疏正则项的次梯度下降
            LL1,bL1 = self.iter_L1(g_k_old, bL1, sparsity, mu)
            g_k = g_k_temp + LL1
            del LL1

            norm_diff = np.linalg.norm(g_k - g_k_old)
            if norm_diff < ep:
                break
            else:
                g_k_old = g_k.copy()
        print(f'迭代次数为{i},误差为{norm_diff}')
        g_k[g_k<0] = 0
        
        psf_init = np.ones((10, 10)) / 25
        x = cv.deconvblind(g_k,psf_init,iterations=30)
        return x, g_k




