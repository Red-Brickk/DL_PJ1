import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def augment(img):
    """
    img: 输入一张28*28的单张图，类型是numpy array，像素值在[0,1]之间。
    返回增强后的图片。
    """
    # 随机旋转：角度范围 -10° 到 +10°
    angle = np.random.uniform(-10, 10)
    img = scipy.ndimage.rotate(img.reshape(28, 28), angle, reshape=False, order=1, mode='constant', cval=0.0)

    # 随机平移：上下左右移动 -2 到 2 个像素
    shift = np.random.uniform(-2, 2, size=2)
    img = scipy.ndimage.shift(img, shift, order=1, mode='constant', cval=0.0)

    # 拉平回向量
    img = img.flatten()

    return img


def visualize_linear_params(W, b):
    """
    可视化 Linear 层的参数：W（权重矩阵）和 b（偏置向量）
    参数：
    W : np.ndarray, shape = [in_dim, out_dim]
    b : np.ndarray, shape = [1, out_dim]
    """
    plt.figure(figsize=(12, 5))

    # 画 W 的热力图
    plt.subplot(1, 2, 1)
    plt.title("Weight Matrix W (Heatmap)")
    plt.imshow(W, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Output Dimension')
    plt.ylabel('Input Dimension')

    # 画 b 的柱状图
    plt.subplot(1, 2, 2)
    plt.title("Bias Vector b (Bar Plot)")
    plt.bar(np.arange(b.shape[1]), b.flatten())
    plt.xlabel('Output Dimension')
    plt.ylabel('Bias Value')

    plt.tight_layout()
    plt.show()

def pad_imgs(imgs, target_size=32):
    """
    把 (50000, 784) 的图片补零成 (50000, 1024)
    """
    num_imgs = imgs.shape[0]
    # 先 reshape 回 (batch_size, 28, 28)
    imgs = imgs.reshape(num_imgs, 28, 28)

    # pad，每边补2个0
    imgs_padded = np.pad(imgs, ((0,0), (2,2), (2,2)), mode='constant', constant_values=0)

    # flatten 回 (batch_size, 1024)
    imgs_padded = imgs_padded.reshape(num_imgs, target_size*target_size)

    return imgs_padded