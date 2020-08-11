import numpy as np
import cv2
import matplotlib.pyplot as plt



pikachu = cv2.imread('pikachu.png', cv2.IMREAD_UNCHANGED)
dim =(256,256) 
pikachu = cv2.resize(pikachu, dim, interpolation = cv2.INTER_AREA)
pikachu = cv2.cvtColor(pikachu, cv2.COLOR_BGR2GRAY)
print(pikachu.shape)
cv2.imwrite('greypikachu.png',pikachu)
mewoth = cv2.imread('mewoth.png', cv2.IMREAD_UNCHANGED)
mewoth = cv2.resize(mewoth, dim, interpolation = cv2.INTER_AREA)
mewoth = cv2.cvtColor(mewoth, cv2.COLOR_BGR2GRAY)

cv2.imwrite('greymewoth.png',mewoth)


box = np.ones((3,3),dtype=int)/9
laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

row = 256 + (box.shape[0]-1)*2
dim = (row,row)
def pad(img,dim):
    matrix = np.zeros((dim), dtype='int')
    
    matrix[2:dim[0]-2,2:dim[0]-2] = img
    matrix.astype(int)
    return matrix

pikachu = pad(pikachu,dim)
mewoth = pad(mewoth,dim)

def masking(img,mask,dim):
    matrix = np.zeros((dim), dtype='int')
    for i in range(1,dim[0]-1):
        for j in range(1,dim[0]-1):
            matrix[i][j] = np.sum(np.multiply(img[i-1:i+2,j-1:j+2],mask))

    matrix = matrix[2:dim[0]-2,2:dim[0]-2] 
    matrix = (matrix* 255)/np.max(matrix)
    matrix = matrix.astype(int) 
    print(matrix.shape)
    return matrix    
    
pikachu = masking(pikachu,box,dim)
mewoth = masking(mewoth,laplace,dim)

cv2.imwrite('blurpikachu.png',pikachu)
cv2.imwrite('blurmeowth.png',mewoth)
hybrid = pikachu + mewoth
#hybrid = cv2.cvtColor(hybrid, cv2.COLOR_BGR2GRAY)
cv2.imwrite('hybrid.png',hybrid)
img = cv2.imread('hybrid.png', cv2.IMREAD_UNCHANGED)

cv2.imshow("New Hybrid image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

     
    