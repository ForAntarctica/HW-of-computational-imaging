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
from joblib import Parallel, delayed

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

def initialize_model(N):
    """
    随机初始化三维模型。

    参数:
        N (int): 三维模型的尺寸（N x N x N）

    返回:
        model (np.array): 初始化的三维模型
    """
    # 随机噪声初始化
    model = np.random.rand(N, N, N)
    return model

def initialize_discrete_angles(num_angles):
    """
    初始化离散化的取向空间，生成一组均匀分布的旋转矩阵。

    参数:
        num_angles (int): 离散取向的数量

    返回:
        angle_discrete (list): 包含 num_angles 个 3x3 旋转矩阵的列表
    """
    angle_discrete = []
    # 使用均匀球面采样（例如，通过随机旋转生成）
    for _ in range(num_angles):
        rot = R.random().as_matrix()
        angle_discrete.append(rot)
    return angle_discrete


def project_model(model, rotation_matrix, N=64):
    """
    将三维模型投影到二维平面，给定一个旋转矩阵。

    参数:
        model (np.array): 三维模型，形状为 (N, N, N)
        rotation_matrix (np.array): 3x3旋转矩阵
        N (int): 投影图像的尺寸

    返回:
        projection (np.array): 2D投影图像，形状为 (N, N)
    """
    # 旋转三维模型
    rotated_model = scipy.ndimage.affine_transform(
        model,
        rotation_matrix,
        order=1,
        mode='constant',
        cval=0.0,
        output_shape=(N, N, N)
    )

    # 投影到XY平面（沿Z轴）
    projection = rotated_model.sum(axis=2)

    # 归一化
    projection = (projection - projection.min()) / (projection.max() - projection.min())
    return projection

def backproject(projection, rotation_matrix, N=64):
    """
    将二维投影图像反投影到三维傅里叶空间，给定一个旋转矩阵。

    参数:
        projection (np.array): 2D投影图像，形状为 (N, N)
        rotation_matrix (np.array): 3x3旋转矩阵
        N (int): 三维模型的尺寸

    返回:
        F3_rotated (np.array): 3D傅里叶空间中的旋转后数据，形状为 (N, N, N)
    """
    # 2D傅里叶变换
    F2 = scipy.fftpack.fftshift(scipy.fftpack.fft2(projection))
    
    # 插入到3D傅里叶空间的z=0平面
    F3 = np.zeros((N, N, N), dtype=np.complex64)
    F3[:, :, N//2] = F2

    # 应用三维旋转（使用旋转矩阵的逆）
    rotated_F3 = scipy.ndimage.affine_transform(
        F3,
        np.linalg.inv(rotation_matrix),
        order=1,
        mode='constant',
        cval=0.0,
        output_shape=(N, N, N)
    )

    return rotated_F3


def compute_similarity(proj_image, model_projection):
    """
    计算投影图像与模型投影的相似度（归一化交叉相关）。

    参数:
        proj_image (np.array): 2D投影图像，形状为 (N, N)
        model_projection (np.array): 三维模型对应取向的2D投影图像，形状为 (N, N)

    返回:
        similarity (float): 相似度得分
    """
    # 归一化
    proj_norm = (proj_image - np.mean(proj_image)) / (np.std(proj_image) + 1e-8)
    model_norm = (model_projection - np.mean(model_projection)) / (np.std(model_projection) + 1e-8)

    # 计算相关系数
    similarity = np.sum(proj_norm * model_norm) / (proj_norm.size)
    return similarity


def E_step(projections, model, angle_discrete, N=64):
    """
    EM算法的E步：计算每个投影对应每个可能取向的概率。

    参数:
        projections (np.array): 3D投影图像数组，形状为 (N_images, 64, 64)
        model (np.array): 当前的三维模型，形状为 (N, N, N)
        angle_discrete (list): 离散化的取向列表（旋转矩阵）
        N (int): 投影图像的尺寸

    返回:
        responsibilities (np.array): 形状为 (N_images, N_angles) 的概率矩阵
    """
    N_images = projections.shape[0]
    N_angles = len(angle_discrete)
    responsibilities = np.zeros((N_images, N_angles))

    # 串行计算所有可能取向的模型投影
    print("E-Step: Projecting all orientations...")
    model_projections = []
    for j in range(N_angles):
        proj = project_model(model, angle_discrete[j], N)
        model_projections.append(proj)
    model_projections = np.array(model_projections)  # (N_angles, N, N)

    # 计算相似度
    print("E-Step: Computing similarities...")
    for i in tqdm(range(N_images), desc="E-Step: Processing Projections"):
        for j in range(N_angles):
            responsibilities[i, j] = compute_similarity(projections[i], model_projections[j])

    # 防止数值问题，添加一个小常数
    responsibilities += 1e-8

    # 归一化为概率
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


def M_step(projections, responsibilities, angle_discrete, N=64):
    """
    执行EM算法的M步，更新三维模型。

    参数:
        projections (np.array): 3D投影图像数组，形状为 (N_images, 64, 64)
        responsibilities (np.array): 形状为 (N_images, N_angles) 的概率矩阵
        angle_discrete (list): 离散化的取向列表（旋转矩阵）
        N (int): 三维模型的尺寸

    返回:
        new_model (np.array): 更新后的三维模型，形状为 (N, N, N)
    """
    fourier_3d = np.zeros((N, N, N), dtype=np.complex64)
    count = np.zeros((N, N, N), dtype=np.float32)

    print("M-Step: Updating 3D Fourier space...")
    for i in tqdm(range(projections.shape[0]), desc="M-Step: Processing Projections"):
        proj = projections[i]
        R_weights = responsibilities[i, :]  # (N_angles,)

        for j, weight in enumerate(R_weights):
            if weight < 1e-6:
                continue  # 忽略权重非常小的情况
            angle = angle_discrete[j]

            # 反投影到三维傅里叶空间
            rotated_F3 = backproject(proj, angle, N)

            # 累加到全局傅里叶空间
            fourier_3d += weight * rotated_F3
            count += weight * (rotated_F3 != 0).astype(float)

    # 防止除以零
    non_zero = count > 0
    fourier_3d[non_zero] /= count[non_zero]

    # 逆傅里叶变换得到新的三维模型
    print("M-Step: Performing inverse Fourier transform...")
    reconstructed_volume = np.real(scipy.fftpack.ifftn(scipy.fftpack.ifftshift(fourier_3d)))

    # 归一化
    reconstructed_volume = (reconstructed_volume - reconstructed_volume.min()) / (reconstructed_volume.max() - reconstructed_volume.min())
    return reconstructed_volume


def EM_algorithm(projections, angle_discrete, N=64, max_iterations=10, tol=1e-4):
    """
    执行EM算法进行角度估计和三维重构。

    参数:
        projections (np.array): 3D投影图像数组，形状为 (N_images, 64, 64)
        angle_discrete (list): 离散化的取向列表（旋转矩阵）
        N (int): 三维模型的尺寸
        max_iterations (int): 最大迭代次数
        tol (float): 收敛阈值（模型变化小于此值时停止迭代）

    返回:
        model (np.array): 重构的三维模型，形状为 (N, N, N)
        responsibilities (np.array): 最终的角度概率矩阵，形状为 (N_images, N_angles)
    """
    # 初始化模型
    model = initialize_model(N)

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        # E步
        responsibilities = E_step(projections, model, angle_discrete, N)
        # M步
        new_model = M_step(projections, responsibilities, angle_discrete, N)
        # 检查收敛
        model_change = np.linalg.norm(new_model - model) / np.linalg.norm(model)
        print(f"Model change: {model_change}")
        if model_change < tol:
            print("Convergence reached.")
            break
        model = new_model

    return model, responsibilities


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

    # 定义离散化的取向（例如，使用均匀采样的球面点）
    # 这里为了简化，使用随机生成的旋转矩阵列表
    # 可以使用更高级的取向采样方法，如HEALPix
    num_angles = 20  # 例如，20个离散取向
    angle_discrete = initialize_discrete_angles(num_angles)
    print(f"定义了 {num_angles} 个离散取向。")

    # 执行EM算法
    print("开始EM算法进行角度估计和三维重构...")
    reconstructed_model, responsibilities = EM_algorithm(
        projections,
        angle_discrete,
        N=64,
        max_iterations=100,
        tol=1e-5
    )
    print("EM算法完成。")

    # 可视化按照Z方向切片
    save_folder = "output/z_slices"
    print(f"Generating {5} XY slices along Z-axis...")
    plot_z_slices(reconstructed_model, num_slices=5, save_folder=save_folder)
    print("All slices have been generated and saved.")


if __name__ == "__main__":
    main()

