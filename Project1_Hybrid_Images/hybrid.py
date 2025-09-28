import sys
import cv2
import numpy as np
import os

def mean_center_sequence(n : int) -> np.ndarray:
    return (np.arange(n, dtype=np.float32) - (n - 1) / 2.0)

def gaussian_blur_kernel_2d(sigma, height, width):
    mcsY = mean_center_sequence(height)
    mcsX = mean_center_sequence(width)
    kernel = np.zeros((height, width), np.float32)
    for y in range(height):
        for x in range(width):
            ySq = mcsY[y]*mcsY[y]
            xSq = mcsY[x]*mcsY[x]
            kernel[y, x] = 1/(2*np.pi*(sigma*sigma)) * np.exp(-(xSq + ySq)/(2*(sigma*sigma)))
    kernel /= np.sum(kernel)
    return kernel

def cross_correlation_2d(img : np.ndarray, kernel : np.ndarray):
    paddedH, paddedW = kernel.shape[0] // 2, kernel.shape[1] // 2
    res = np.zeros(img.shape, dtype=np.float32)

    if (len(res.shape) == 2):
        paddedImg = np.pad(img,
                            ((paddedH, paddedH), (paddedW, paddedW)),
                             mode='constant')
        imageHeight, imageWidth = paddedImg.shape[0], paddedImg.shape[1]
        # 이렇게 하는 이유는 만paddedH약 i, j가 끝까지 간다면 region은 커널의 절반만큼 짤리게 됨
        hStart, hEnd = paddedH, imageHeight - paddedH
        wStart, wEnd = paddedW, imageWidth - paddedW
        for i in range(hStart, hEnd):
            for j in range(wStart, wEnd):
                region = paddedImg[ i - paddedH: i + paddedH + 1,
                                    j - paddedW: j + paddedW + 1]
                res[i - paddedH, j - paddedW] = np.sum(region * kernel)
    elif (len(res.shape) == 3):
        paddedImg = np.pad(img,
                             ((paddedH, paddedH), (paddedW, paddedW), (0, 0)),
                             mode='constant')
        imageHeight, imageWidth = paddedImg.shape[0], paddedImg.shape[1]
        # 이렇게 하는 이유는 만약 i, j가 끝까지 간다면 region은 커널의 절반만큼 짤리게 됨
        hStart, hEnd = paddedH, imageHeight - paddedH
        wStart, wEnd = paddedW, imageWidth - paddedW
        for channel in range(0, 3):
            for i in range(hStart, hEnd):
                for j in range(wStart, wEnd):
                    region = paddedImg[ i - paddedH: i + paddedH + 1,
                                        j - paddedW: j + paddedW + 1,
                                        channel]
                    res[i - paddedH, j - paddedW, channel] = np.sum(region * kernel)

    return res

def convolve_2d(img : np.ndarray , kernel : np.ndarray):
    return cross_correlation_2d(img, np.flip(kernel, axis=(0,1)))

def low_pass(img, sigma, size):
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size, sharpness_weight):
    lowPassedImage = low_pass(img, sigma, size)
    return np.add(img, np.multiply((img - lowPassedImage), sharpness_weight)).astype(np.float32)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor, sharpness_weight):
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
        img1 = high_pass(img1, sigma1, size1, sharpness_weight)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2, sharpness_weight)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)