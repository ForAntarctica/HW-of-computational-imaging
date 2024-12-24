import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read data
pth = './noisy_photos/1.jpg'
img = plt.imread(pth).astype(np.float32) / 255
print(img.shape, img.dtype)

# Plot image
plt.imshow(img)
plt.show()
plt.close()

# Gaussian Filter
filtered_img = cv2.GaussianBlur(img, (3, 3), 1)
plt.imshow(filtered_img)
plt.show()
plt.close()