
import cv2
import numpy as np
 
# Load the image in greyscale
img = cv2.imread('north.jpg',0)
cv2.imshow('Original image',img)
cv2.waitKey(0) 
# Apply Gaussian Blur
blur = cv2.GaussianBlur(img,(3,3),0)
 
# Apply Laplacian operator in some higher datatype
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
laplacian1 = laplacian/laplacian.max()
log = laplacian1 
cv2.imshow('a7',laplacian1)
cv2.waitKey(0)
def Zero_crossing(image,val):
    z_c_image = np.zeros(image.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1


            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i,j]>val:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<val:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)
    

    return z_c_image
img1 = Zero_crossing(log,0)
print(img1)
img2 = Zero_crossing(log,8)
print(img2)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey()