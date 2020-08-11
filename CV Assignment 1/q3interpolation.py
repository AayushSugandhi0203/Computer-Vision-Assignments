import cv2
import numpy as np
import math

def nearestinterpolation(img):
    print("Generating Nearest Interpolate Image")
    img_nearest_interpolate = np.zeros((512,512),dtype = int)
    x = 0
    y = 0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img_nearest_interpolate[x][y] = img[i][j]
            img_nearest_interpolate[x][y+1] = img[i][j]
            y = y + 2
            if y>=512:
                y = 0
                
        
        x =x +2
    x = 0
    y = 0
    
              
    for i in range(0,512,2):
        for j in range(0,512):
            img_nearest_interpolate[i+1][j] = img_nearest_interpolate[i][j]   
            #img_nearest_interpolate[i][j+1] = img_nearest_interpolate[x][y+1]
        
        
    cv2.imwrite('img_nearest_interpolate.tif',img_nearest_interpolate)
    
    
def bilinearinterpolation(resizedImg):
    print("Generating Bilinear Interpolate Image")
    row = resizedImg.shape[0]
    col = resizedImg.shape[1]
    newRow = 512
    newCol = 512
    rowRatio = int(newRow/row)
    colRatio = int(newCol/col)
    newImg = np.zeros((newRow,newCol))

    for i in range(0,newImg.shape[0],rowRatio):
        for j in range(0,newImg.shape[1],colRatio):
            newImg[i,j] = resizedImg[int(i/rowRatio),int(j/colRatio)] #putting original pixels in new img

    for i in range(0,newImg.shape[0],rowRatio):
        for j in range(1,newImg.shape[1]-1,colRatio):
            newImg[i,j] = 0.5*newImg[i,j-1]+0.5*newImg[i,j+1]

    for i in range(1,newImg.shape[0]-1,rowRatio):
        for j in range(0,newImg.shape[1],colRatio):
            newImg[i,j] = 0.5*newImg[i-1,j]+0.5*newImg[i+1,j]

    for i in range(1,newImg.shape[0]-1,rowRatio):
        for j in range(1,newImg.shape[1]-1,colRatio):
            newImg[i,j] = 0.5*newImg[i,j-1]+0.5*newImg[i,j+1]
    
    cv2.imwrite('img_bilinear_interpolate.tif',newImg)
    img = cv2.imread('img_bilinear_interpolate.tif', cv2.IMREAD_UNCHANGED)
            

def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Paddnig
def padding(img,H,W):
    zimg = np.zeros((H+4,W+4))
    zimg[2:H+2,2:W+2] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2]=img[:,0:1]
    zimg[H+2:H+4,2:W+2]=img[H-1:H,:]
    zimg[2:H+2,W+2:W+4]=img[:,W-1:W]
    zimg[0:2,2:W+2]=img[0:1,:]
    #Pad the missing eight points
    zimg[0:2,0:2]=img[0,0]
    zimg[H+2:H+4,0:2]=img[H-1,0]
    zimg[H+2:H+4,W+2:W+4]=img[H-1,W-1]
    zimg[0:2,W+2:W+4]=img[0,W-1]
    return zimg


def bicubic(img, ratio, a):
    #Get image size
    print("Generating Bicubic Interpolate Image")
    H,W = img.shape

    img = padding(img,H,W)
    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW))

    h = 1/ratio

    
    for j in range(dH):
        for i in range(dW):
            x, y = i * h + 2 , j * h + 2

            x1 = 1 + x - math.floor(x)
            x2 = x - math.floor(x)
            x3 = math.floor(x) + 1 - x
            x4 = math.floor(x) + 2 - x

            y1 = 1 + y - math.floor(y)
            y2 = y - math.floor(y)
            y3 = math.floor(y) + 1 - y
            y4 = math.floor(y) + 2 - y

            mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
            mat_m = np.matrix([[img[int(y-y1),int(x-x1)],img[int(y-y2),int(x-x1)],img[int(y+y3),int(x-x1)],img[int(y+y4),int(x-x1)]],
                                [img[int(y-y1),int(x-x2)],img[int(y-y2),int(x-x2)],img[int(y+y3),int(x-x2)],img[int(y+y4),int(x-x2)]],
                                [img[int(y-y1),int(x+x3)],img[int(y-y2),int(x+x3)],img[int(y+y3),int(x+x3)],img[int(y+y4),int(x+x3)]],
                                [img[int(y-y1),int(x+x4)],img[int(y-y2),int(x+x4)],img[int(y+y3),int(x+x4)],img[int(y+y4),int(x+x4)]]])
            mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
            dst[j, i] = np.dot(np.dot(mat_l, mat_m),mat_r)

               
    return dst
def bicubicinterpolation(resizedImg):
    ratio = 2
# Coefficient
    a = -1/2

    dst = bicubic(resizedImg, ratio, a)       
    cv2.imwrite('img_bicubic_interpolate.tif',dst)
    print("Go and check folder")
    


img = cv2.imread('reduce_lena_256_matrix.tif', cv2.IMREAD_UNCHANGED)



nearestinterpolation(img)
bilinearinterpolation(img)
bicubicinterpolation(img)