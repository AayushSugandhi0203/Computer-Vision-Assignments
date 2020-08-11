import cv2
import numpy as np
import math


def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    #img2 = np.array(img2)/255.
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return mse ,10 * math.log10(1. / mse)

img1 = cv2.imread('lena_gray_512.tif', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('img_nearest_interpolate.tif', cv2.IMREAD_UNCHANGED)
diff = img1 - img2
print("Nearest Interpolate")
mse , psnr = compute_psnr(img1,img2)
print("MSE value is",mse)
print("PSNR value is",psnr)
cv2.imshow("New image is", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread('img_bilinear_interpolate.tif', cv2.IMREAD_UNCHANGED)
diff = img1 - img2
print("Bilinear")
mse , psnr = compute_psnr(img1,img2)
print("MSE value is",mse)
print("PSNR value is",psnr)
cv2.imshow("New image is", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread('img_bicubic_interpolate.tif', cv2.IMREAD_UNCHANGED)
diff = img1 - img2
print("Bicubic")
mse , psnr = compute_psnr(img1,img2)
print("MSE value is",mse)
print("PSNR value is",psnr)
cv2.imshow("New image is", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

