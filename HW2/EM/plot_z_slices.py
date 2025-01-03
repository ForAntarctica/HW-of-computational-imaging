import os
import numpy as np
import matplotlib.pyplot as plt

def plot_z_slices(volume, num_slices=5, save_folder="z_slices"):
    """
    按照Z方向切片，生成指定数量的XY平面的灰度图并保存。
    
    参数：
    - volume: 3D numpy数组表示重构的三维结构
    - num_slices: 生成的切片数量
    - save_folder: 切片图像保存的文件夹
    """
    N = volume.shape[0]
    indices = np.linspace(0, N-1, num_slices, dtype=int)
    
    # 创建保存文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for i, z in enumerate(indices):
        plt.figure(figsize=(6, 6))
        plt.imshow(volume[z, :, :], cmap='gray')
        plt.title(f'XY Slice at Z={z}')
        plt.axis('off')
        slice_filename = os.path.join(save_folder, f'xy_slice_z{z}.png')
        plt.savefig(slice_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Saved slice {i+1} at Z={z} as {slice_filename}')