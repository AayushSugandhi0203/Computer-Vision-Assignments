import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy
from scipy import signal
import matplotlib.pyplot as plt

lena = cv2.imread('lena.bmp', cv2.IMREAD_UNCHANGED)
dim  = (256,256)
lena = cv2.resize(lena, dim, interpolation = cv2.INTER_AREA)
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
cv2.imwrite('greylena.bmp',lena)



f = np.fft.fft2(lena)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(lena, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()



def truncation(img,dim):
    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2
    val  = int(((rows*(dim))//100)//2)
    fshift[crow-val:crow+val, ccol-val:ccol+val] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    plt.show()
    
truncation(lena,25)
truncation(lena,12)    
truncation(lena,6.25)   

iris = cv2.imread('iris-illustration.bmp', cv2.IMREAD_UNCHANGED)
dim  = (256,256)
iris = cv2.resize(iris, dim, interpolation = cv2.INTER_AREA)
iris = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
cv2.imwrite('greyiris.bmp',iris) 

f = np.fft.fft2(iris)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(iris, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

def truncation(img,dim):
    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2
    val  = int(((rows*(dim))//100)//2)
    fshift[crow-val:crow+val, ccol-val:ccol+val] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    plt.show()
    
truncation(iris,25)
truncation(iris,12)    
truncation(iris,6.25)     

def inverse(img):
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
    
invlena = inverse(lena) 
inviris = inverse(iris)   

def compute_snr(img1,img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    #img2 = np.array(img2)/255.
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return mse ,10 * math.log10(1. / mse)

img1 = lena
img2 = invlena


mse , psnr = compute_snr(img1,img2)

print("PSNR value is",psnr)

mse , psnr = compute_snr(iris,inviris)

print("PSNR value is",psnr)

