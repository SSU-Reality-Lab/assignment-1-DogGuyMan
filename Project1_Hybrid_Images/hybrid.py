import sys
import cv2
import numpy as np
import os

def mean_center_sequence(n : int) -> np.ndarray:
    return (np.arange(n, dtype=np.float32) - (n - 1) / 2.0)

def gaussian_blur_kernel_2d(sigma, height, width):
    mcsY = mean_center_sequence(height)
    mcsY.shape = (1, height)
    mcsY = mcsY.T

    mcsX = mean_center_sequence(width)
    mcsX.shape = (1, width)

    gY = np.exp(-(mcsY*mcsY) / (2.0 * sigma*sigma))
    gY /= gY.sum()
    gX = np.exp(-(mcsX*mcsX) / (2.0 * sigma*sigma))
    gX /= gX.sum()

    return np.outer(gY, gX)  # Return the calculated Gaussian filter

'''
픽셀의 위치와 이미지의 shape를 비교하여 bound인지 아닌지 리턴
'''
def check_is_bound(shape : tuple, pixel_pos : tuple) -> bool:
    if pixel_pos[1] < 0 or pixel_pos[0] < 0:
        return False
    if pixel_pos[1] >= shape[1] or pixel_pos[0] >= shape[0]:
        return False
    return True

# ✅ numpy로 최적화 하기
def cross_correlation_2d(img : np.ndarray , kernel : np.ndarray):
    # print(img.shape) # 쉐이프 검사 파이썬도 y, x 구나.
    # print(kernel.shape) # 쉐이프 검사 파이썬도 y, x 구나.
    image_height, image_width = img.shape
    kernel_height, kernel_width = kernel.shape
    kernel_y_mean, kernel_x_mean = kernel_height // 2, kernel_width // 2
    res = np.zeros(img.shape)
    comp_res = np.zeros(img.shape)

    # print(kernel_y_mean, kernel_x_mean) # 평균 검사
    for i in range(image_height):
        for j in range(image_width):
            for u in range(kernel_height):
                for v in range(kernel_width):
                    # print(i, j, (u - kernel_y_mean), (v - kernel_x_mean)) # 픽셀과 오프셋
                    offseted_pixel = ((i + u - kernel_y_mean), (j + v - kernel_x_mean))
                    if check_is_bound(img.shape, (offseted_pixel[1], offseted_pixel[0])):
                        try :
                            res[i][j] += kernel[u][v] * img[offseted_pixel[1]][offseted_pixel[0]]
                        except :
                            print("cross_correlation_2d error")

    return res

def convolve_2d(img : np.ndarray , kernel : np.ndarray):
    return cross_correlation_2d(img, np.flip(kernel))

def low_pass(img, sigma, size):
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    mask = low_pass(img, sigma, size)
    return (img * 1.5) - (mask * 0.5)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

#gaussian_blur_kernel_2d(1, 3, 7)
# imp = np.zeros((7, 7), np.float32)
# imp[7 // 2, 7 // 2] = 1.0
# print("imp")
# print(imp)
# kernel = np.array([
#     [0, 1.5, 0],
#     [8.5, 1, 2.5],
#     [7.5, 1, 3.5],
#     [6.5, 1, 4.5],
#     [0, 5.5, 0]
# ])
# print("kernel")
# print(kernel)
#
# # print(cross_correlation_2d(imp, kernel))
# # print(scipy.signal.correlate(imp, kernel, mode='same'))
# #
# # print(convolve_2d(imp, kernel))
# # print(scipy.signal.convolve2d(imp, kernel, mode='same'))
#
# print(convolve_2d(imp, gaussian_blur_kernel_2d(1, 7, 7)).round(3))
# print(cv2.GaussianBlur(imp, (7, 7), 1, borderType=cv2.BORDER_CONSTANT).round(3))