import numpy as np
import cv2
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA - imageB) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	print(err)
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

imageA = 'lena_gray_256.tif'
imageB = 'resized_lena_256.tif'
imgA = cv2.imread('lena_gray_256.tif', cv2.IMREAD_UNCHANGED)
imgB = cv2.imread('resized_lena_256.tif', cv2.IMREAD_UNCHANGED)
cv2.imshow("t",imgA)
cv2.waitKey(0)
cv2.destroyAllWindows()
mse(imgA, imgB)