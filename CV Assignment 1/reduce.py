import cv2
 
img = cv2.imread('C:/Users/TCD/Desktop/standard_test_images/lena_gray_512.tif', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size

dim = (256, 256)
# resize image
resized_lena_256 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized_lena_256.shape)
 
cv2.imshow("Resized image", resized_lena_256)
cv2.imwrite('resized_lena_256.tif',resized_lena_256)
cv2.waitKey(0)
cv2.destroyAllWindows()