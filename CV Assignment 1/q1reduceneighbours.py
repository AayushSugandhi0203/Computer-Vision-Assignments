import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lena_gray_512.tif', cv2.IMREAD_UNCHANGED)
cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgarr = np.array(img)
x = 0
y = 0
imgnew = np.zeros((256,256),dtype = int)
print(imgnew)
print(imgarr.shape)
print(imgarr.ndim)
print(imgarr)
for i in range(0,imgarr.shape[0]):
    for j in range(0,imgarr.shape[1]):
        if i%2==0 and j%2==0:
            
            x = i//2
            y = j//2
            imgnew[x][y]=min(imgarr[i][j],imgarr[i+1][j],imgarr[i-1][j],imgarr[i][j-1],imgarr[i][j+1])
            
            

print(imgnew.shape)            
for i in range(0,imgnew.shape[0]):
    for j in range(0,imgnew.shape[1]):
        
        imgnew[i][j] = int(round(imgnew[i][j]))  
        #print(imgnew[i][j])          
            
print(imgnew)  
print(imgnew.shape)
print(imgnew.ndim)  
   
plt.imshow(imgnew, cmap="gray")
plt.show()  
print("Sucees")   
cv2.imwrite('reduce_lena_256_matrix.tif',imgnew)
img = cv2.imread('reduce_lena_256_matrix.tif', cv2.IMREAD_UNCHANGED)
cv2.imshow("New image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

