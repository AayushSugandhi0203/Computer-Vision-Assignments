import cv2
import numpy as np
from matplotlib import pyplot as plt

def fourier(img):
    f  =np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back  = np.abs(img_back)
    f_bounded = 50 * np.log(img_back)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    
    plt.imshow(f_img, cmap = 'gray')
    plt.show()
    return f_img

mewoth = cv2.imread('greymewoth.png', cv2.IMREAD_UNCHANGED)
pikachu = cv2.imread('greypikachu.png', cv2.IMREAD_UNCHANGED)

pikachu_fourier = fourier(pikachu)
mewoth_fourier = fourier(mewoth)


hybrid =   pikachu_fourier+mewoth_fourier
plt.imshow(hybrid, cmap = 'gray')
plt.show()
cv2.imwrite('hybridfourier.png',hybrid)