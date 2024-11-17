from PIL import Image
import numpy as np

class ImageProcessor:
    def __init__(self, image_path, num_images):
        self.image_path = image_path
        self.num_images = num_images
        self.clean_images = []
        self.noisy_images = []
        
    def read_images(self):
        for i in range(self.num_images):
            image = Image.open(f'{self.image_path}/{i}.jpg')
            self.clean_images.append(image)

    def add_salt_pepper_noise(self,index_image , salt_prob, pepper_prob):
        """添加椒盐噪声"""
        image_array = np.array(self.clean_images[index_image])
        noisy_image = image_array.copy()
        total_pixels = image_array.size // image_array.shape[2]
        num_salt = np.ceil(salt_prob * total_pixels)
        num_pepper = np.ceil(pepper_prob * total_pixels)
        # 添加盐噪声
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
        noisy_image[coords[0], coords[1], :] = 255

        # 添加胡椒噪声
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
        noisy_image[coords[0], coords[1], :] = 0

        self.noisy_images.append(Image.fromarray(noisy_image))

    def add_gaussian_noise(self, index_image, mean, sigma):
        """添加高斯噪声"""
        image_array = np.array(self.clean_images[index_image])
        gaussian = np.random.normal(mean, sigma, image_array.shape)
        noisy_image = image_array + gaussian
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8) 
        self.noisy_images.append(Image.fromarray(noisy_image))

    def add_poisson_noise(self, index_image):
        """添加泊松噪声"""
        image_array = np.array(self.clean_images[index_image])
        vals = len(np.unique(image_array))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image_array * vals) / float(vals)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8) 
        self.noisy_images.append(Image.fromarray(noisy_image))

    def add_all_noises(self, index_image, salt_prob, pepper_prob, mean, sigma):
        image_array = np.array(self.clean_images[index_image])
        vals = len(np.unique(image_array))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image_array * vals) / float(vals)


        gaussian = np.random.normal(mean, sigma, image_array.shape)
        noisy_image = noisy_image + gaussian

        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8) 

        total_pixels = image_array.size // image_array.shape[2]
        num_salt = np.ceil(salt_prob * total_pixels)
        num_pepper = np.ceil(pepper_prob * total_pixels)
        # 添加盐噪声
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
        noisy_image[coords[0], coords[1], :] = 255

        # 添加胡椒噪声
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
        noisy_image[coords[0], coords[1], :] = 0

        self.noisy_images.append(Image.fromarray(noisy_image))

    def PGD_denoising(self, index_image,lam,t,ep):
        # 近段梯度下降去噪点函数
        image_array = np.array(self.noisy_images[index_image],dtype=np.float64) 
        # 这里正则项选择L1范数，保真项为||y-x||^2
        image_array_k_old = image_array.copy()
        image_array_k = image_array.copy()
        for i in range(1, 100000):
            # 梯度下降
            image_array_k_temp = image_array_k_old - t * (image_array_k_old - image_array)
            # 软阈值函数
            for x, row in enumerate(image_array_k_temp):
                for y, pixel in enumerate(row):
                    for z, channel in enumerate(pixel):
                        if channel > lam * t:
                            image_array_k[x][y][z] = channel - lam * t
                        elif abs(channel) < lam * t:
                            image_array_k[x][y][z] = 0
                        elif channel < -lam * t:
                            image_array_k[x][y][z] = channel + lam * t    
            norm_diff = np.linalg.norm(image_array_k - image_array_k_old)                           
            if norm_diff < ep:
                break
            else:
                image_array_k_old = image_array_k.copy()
        print(f'迭代次数为{i},误差为{norm_diff}')
        return Image.fromarray(image_array_k.astype(np.uint8))
                        
    def ADMM_denoising(self, index_image, lam, t1, t2, alpha_k):
        # ADMM去噪点函数
        image_array = np.array(self.noisy_images[index_image],dtype=np.float64)
        # 这里正则项选择L1范数，保真项为||y-x||^2
        image_array_k_old = np.ones_like(image_array)
        u_k_old = np.zeros_like(image_array)
        s_k_old = np.zeros_like(image_array)
        u_k = np.zeros_like(image_array)
        for i in range(1, 2000):
            # 更新image_array
            image_array_k = (1 + 2*t2)**(-1) * (image_array + t2 * u_k_old - s_k_old) 

            # 更新u_k
            u_k_temp = u_k_old + t1 * t2 * (image_array_k - u_k_old + (1/t2) * s_k_old)
            for x, row in enumerate(u_k_temp):
                for y, pixel in enumerate(row):
                    for z, channel in enumerate(pixel):
                        if channel > lam * t1:
                            u_k[x][y][z] = channel - lam * t1
                        elif abs(channel) < lam * t1:
                            u_k[x][y][z] = 0
                        elif channel < -lam * t1:
                            u_k[x][y][z] = channel + lam * t1

            # 更新s_k
            s_k = s_k_old + alpha_k * (image_array_k - u_k)
            norm_diff = np.linalg.norm(image_array_k - image_array_k_old)
            if norm_diff < 1e-3:
                break    
            else:
                image_array_k_old = image_array_k.copy()
                u_k_old = u_k.copy()
                s_k_old = s_k.copy()
            
        print(f'迭代次数为{i},误差为{norm_diff}')
        return Image.fromarray(image_array_k.astype(np.uint8))


