import numpy as np
import cv2
import matplotlib.pyplot as plt



hybrid = cv2.imread('hybrid.png', cv2.IMREAD_UNCHANGED)
m = 3
n = 3
def guassian(m,n,sigma):
    guass = np.zeros((m,n))
    m = m//2
    n=n//2
    for x in range(-m,m+1):
        for y in range(-n,n+1):
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(x**2 + y**2))/(2*sigma**2)
            guass[x+m,y+n] = x2/x1
    print(guass)        
    return guass

def pad(img,dim):
    matrix = np.zeros((dim), dtype='int')
    
    matrix[2:dim[0]-2,2:dim[0]-2] = img
    matrix.astype(int)
    return matrix



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

g = guassian(3,3,1.67)
dim = (260,260)
hybrid = pad(hybrid,dim)
hybrid = masking(hybrid,g,dim)
cv2.imwrite('guassianhybrid.png',hybrid)
img = cv2.imread('guassianhybrid.png', cv2.IMREAD_UNCHANGED)

cv2.imshow("New Hybrid image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

