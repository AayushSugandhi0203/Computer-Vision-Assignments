import numpy as np
import cv2
import matplotlib.pyplot as plt

pikachu = cv2.imread('greypikachu.png', cv2.IMREAD_UNCHANGED)
mewoth = cv2.imread('greymewoth.png', cv2.IMREAD_UNCHANGED)

mewopika = np.hstack((pikachu[:,:128],mewoth[:,128:]))
cv2.imshow("Mewopika",mewopika)
cv2.waitKey(0)
cv2.destroyAllWindows()

#1 Load the Images
#2 Find Gaussian Pyramid for both images(no. of level = 5)
#3 From guassian find Laplacian Pyramid
#4 Now join the left half of image1 and right half of image2 in each level
#5 Finally from  this joint image pyramids, reconstruct original image


pikachu_copy = pikachu.copy()
guass_pika = [pikachu_copy]

for i in range(0,6):
    pikachu_copy = cv2.pyrDown(pikachu_copy)
    guass_pika.append(pikachu_copy)
    #print("Shape of pikachu is",pikachu_copy.shape)
    
mewoth_copy = mewoth.copy()
guass_mewo = [mewoth_copy]

for i in range(0,6):
    mewoth_copy = cv2.pyrDown(mewoth_copy)
    guass_mewo.append(mewoth_copy)    

print("Guassian Pyramids are")    
for i in range(0,6):
    #pass    
    cv2.imshow("Pikachu"+ str(i), guass_pika[i])
    cv2.imshow("Mewoth"+str(i),guass_mewo[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
pikachu_copy  = guass_pika[5]
lapla_pika = [pikachu_copy]

for i in range(5,0,-1):
    guass_more = cv2.pyrUp(guass_pika[i])
    laplacian = cv2.subtract(guass_pika[i-1],guass_more)
    lapla_pika.append(laplacian)
    
mewoth_copy  = guass_mewo[5]
lapla_mewo = [mewoth_copy]
for i in range(5,0,-1):
    guass_more = cv2.pyrUp(guass_mewo[i])
    laplacian = cv2.subtract(guass_mewo[i-1],guass_more)
    lapla_mewo.append(laplacian)   
    
pyramid = []
i = 0
for pika, mewo in zip(lapla_pika,lapla_mewo):
    i = i + 1
    cols , row = pika.shape
    laplacian =np.hstack((pika[:,:int(cols/2)],mewo[:,int(cols/2):]))   
    pyramid.append(laplacian)

reconstruct = pyramid[0]

for i in range(1,6):
    reconstruct = cv2.pyrUp(reconstruct)
    
    reconstruct = cv2.add(pyramid[i],reconstruct)
             
cv2.imshow("Merges",reconstruct)    
cv2.waitKey(0)
cv2.destroyAllWindows()  

laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
dim = (260,260)

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
    
    return matrix    


def lap(img):
    l  =[]
    for i in img:
        lap_img = pad(img,dim)
        lap_img  = masking(lap_img,laplace,dim)
        l.append(lap_img)
        return reversed(l)

def hybrid(lp1,lp2):
    q = []
    for img1,img2 in zip(lp1,lp2):
        q.append(img1+img2)
    
    q0 = q[0]
    for i in range(1,len(q)):
        q0 =   cv2.pyrUp(q0,q[i])      
        q0 = cv2.add(q0,q[i])
    return q0,q 
   
lp1 = lap(guass_pika[0])
lp2 = lap(guass_mewo[0])

hybd,lsnew =  hybrid(lp1,lp2)

hybd = np.array(hybd)
hybd = hybd.astype(dtype = np.float32)

cv2.imshow("Hybrid",hybd)    
cv2.waitKey(0)
cv2.destroyAllWindows()