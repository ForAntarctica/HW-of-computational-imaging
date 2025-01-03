import os
import numpy as np
import scipy.fft
import scipy.ndimage
import mrcfile
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from skimage.transform import rotate
from skimage.measure import marching_cubes
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from numpy.fft import fftshift, fft2, ifft2, ifftn, fftn, ifftshift
from plot_z_slices import plot_z_slices
from tqdm import tqdm  # 用于显示进度条

def read_mrcs(file_path):

    """
    读取.mrcs文件并返回一个形状为 (N, 64, 64) 的NumPy数组。

    参数:
        file_path (str): .mrcs文件的路径

    返回:
        projections (np.array): 形状为 (N, 64, 64) 的二维投影图像数组
    """
    with mrcfile.open(file_path, permissive=True) as mrc:
        projections = mrc.data.copy()
    return projections

def read_pkl(file_path):
    """
    读取.pkl文件并返回角度矩阵和相对中心偏移。

    参数:
        file_path (str): .pkl文件的路径

    返回:
        Rotations (np.array): 形状为 (N, 3, 3) 的旋转矩阵
        Translations (np.array): 形状为 (N, 2, 2) 的平移矩阵
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    Rotations = data[0]
    Translations = data[1]
    return Rotations, Translations

def reconstruct_3d_optimized(projections, rotations, translations, N):
    """
    优化后的基于傅里叶切片定理的三维重构。
    
    参数：
    - projections: 3D numpy数组，形状为 (N_images, 64, 64)
    - rotations: 3D numpy数组，形状为 (N_images, 3, 3)
    - translations: 3D numpy数组，形状为 (N_images, 2, 2) （本示例中为零）
    - N: 重构体积的尺寸（例如，64）
    
    返回：
    - reconstructed_volume: 3D numpy数组，形状为 (N, N, N)
    """
    N_images = projections.shape[0]
    
    # 初始化三维傅里叶空间和计数矩阵
    fourier_3d = np.zeros((N, N, N), dtype=complex)
    count = np.zeros((N, N, N), dtype=float)
    
    # 生成频率坐标
    freq = np.fft.fftfreq(N).reshape(-1, 1)
    freq = fftshift(freq)  # 频率从负到正
    
    # 预计算频率网格
    u = freq.flatten()
    v = freq.flatten()
    U, V = np.meshgrid(u, v, indexing='ij')
    freq_2d = np.stack((U, V, np.zeros_like(U)), axis=-1)  # (64, 64, 3)
    
    
    # 逐个投影处理
    for i in tqdm(range(N_images), desc="Processing Projections"):
        proj = projections[i]
        R = rotations[i]  # (3, 3)
        T = translations[i]  # (2, 2) — 在本示例中为零
        
        # 处理翻译（偏移量）— 假设为零，若非零需进行相应调整
        # 这里暂时忽略偏移量
        
        # 2D傅里叶变换
        F2 = fftshift(fft2(proj))
        
        # 生成频率二维网格
        F2_flat = F2.flatten()  # (64*64,)
        
        # 应用旋转矩阵
        rotated_freq = freq_2d @ R.T  # (64, 64, 3)
        
        # 映射到三维傅里叶空间索引
        # 将频率范围 [-0.5, 0.5] 映射到索引范围 [0, N-1]
        x = np.round((rotated_freq[:, :, 0] + 0.5) * (N - 1)).astype(int)
        y = np.round((rotated_freq[:, :, 1] + 0.5) * (N - 1)).astype(int)
        z = np.round((rotated_freq[:, :, 2] + 0.5) * (N - 1)).astype(int)
        
        # 创建掩码，确保索引在范围内
        mask = (x >= 0) & (x < N) & (y >= 0) & (y < N) & (z >= 0) & (z < N)
        valid_x = x[mask]
        valid_y = y[mask]
        valid_z = z[mask]
        F2_valid = F2_flat[mask.flatten()]
        
        # 使用线性索引更新三维傅里叶空间
        fourier_3d[valid_x, valid_y, valid_z] += F2_valid
        count[valid_x, valid_y, valid_z] += 1
    
    # 平均重叠的点
    # non_zero = count > 0
    # fourier_3d[non_zero] /= count[non_zero]
    non_zero = count > 0
    fourier_3d[non_zero] /= N_images
    

    # 逆傅里叶变换得到重构的三维体积
    reconstructed_volume = np.abs(ifftn(ifftshift(fourier_3d)))
    # 图像反色
    min_val = reconstructed_volume.min()
    max_val = reconstructed_volume.max()
    normalized_volume = (reconstructed_volume - min_val) / (max_val - min_val)

    reconstructed_volume = 1 - normalized_volume

    return reconstructed_volume

def main():
    # 文件路径（请根据实际路径修改）
    mrcs_file_path = 'data/task1_projections.mrcs'
    pkl_file_path = 'data/task1_poses.pkl'

    # 读取数据
    print("读取.mrcs文件...")
    projections = read_mrcs(mrcs_file_path)
    print(f"读取了 {projections.shape[0]} 个二维投影。")

    print("读取.pkl文件...")
    rotations, translations = read_pkl(pkl_file_path)
    print(f"读取了 {rotations.shape[0]} 个旋转矩阵和 {translations.shape[0]} 个平移矩阵。")

    # 检查数据一致性
    assert projections.shape[0] == rotations.shape[0] == translations.shape[0], "投影数量与角度数量不匹配。"

    # 重构三维体积
    print("开始三维重构...")
    volume = reconstruct_3d_optimized(projections, rotations, translations, N=64)
    print("三维重构完成。")

    # 可视化结果
    # 按照Z方向切片，生成5张XY平面的灰度图
    save_folder = "Fourier_Slice_Theorem/output"
    print(f"Generating {5} XY slices along Z-axis...")
    plot_z_slices(volume, num_slices=5, save_folder=save_folder)
    print("All slices have been generated and saved.")

if __name__ == "__main__":
    main()
