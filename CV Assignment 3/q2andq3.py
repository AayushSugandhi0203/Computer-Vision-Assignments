import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy
pikachu = cv2.imread('pikachu.png', cv2.IMREAD_UNCHANGED)
dim  = (64,64)
pikachu = cv2.resize(pikachu, dim, interpolation = cv2.INTER_AREA)
pikachu = cv2.cvtColor(pikachu, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image",pikachu)
cv2.waitKey(0)
cv2.destroyAllWindows()
def imagepad(img, dim):
    image  = np.lib.pad(pikachu, dim, 'constant', constant_values=(0, 0))
    return image
    
listimage  = []
listimage.append(pikachu)
for i in range(0,4):
    
    new = imagepad(listimage[-1],listimage[-1].shape) 
    listimage.append(new)
def fourier(img):
    

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
pikachu1 = np.zeros((128,128))
pikachu2 = np.zeros((256,256))
pikachu3 = np.zeros((512,512))
for i in range(64):
    for j in range(64):
        pikachu1[i][j] = pikachu[i][j]
        pikachu2[i][j] = pikachu[i][j]
        pikachu3[i][j] = pikachu[i][j]

fourier(pikachu)
fourier(pikachu1)
fourier(pikachu2)
fourier(pikachu3)





img = cv2.imread('noiseball.png',cv2.IMREAD_UNCHANGED)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# calculate amplitude spectrum
mag_spec = 20*np.log(np.abs(fshift))

r = f.shape[0]//2        
c = f.shape[1]//2           
p = 3                         
n = 1                         
fshift2 = np.copy(fshift)


fshift2[0:r-n , c-p:c+p] = 0.001

fshift2[r+n:r+r, c-p:c+p] = 0.001

mag_spec2 = 20*np.log(np.abs(fshift2))
inv_fshift = np.fft.ifftshift(fshift2)

img_recon = np.real(np.fft.ifft2(inv_fshift))




dst = cv2.fastNlMeansDenoising(img)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mag_spec, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(mag_spec2, cmap = 'gray')
plt.title('Magnitude Spectrum after suppression'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.imshow("Image",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

