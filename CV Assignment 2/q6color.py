import numpy as np
import cv2
import matplotlib.pyplot as plt



pikachu = cv2.imread('pikachu.png', cv2.IMREAD_UNCHANGED)
print(pikachu.shape)
pikachu = cv2.cvtColor(pikachu, cv2.COLOR_BGRA2BGR)
dim =(256,256) 
pikachu = cv2.resize(pikachu, dim, interpolation = cv2.INTER_AREA)

print(pikachu.shape)
mewoth = cv2.imread('mewoth.png', cv2.IMREAD_UNCHANGED)
mewoth = cv2.cvtColor(mewoth, cv2.COLOR_BGRA2BGR)
mewoth = cv2.resize(mewoth, dim, interpolation = cv2.INTER_AREA)


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
guassfiltr = guassian(7,7,2)

def Spatialcolor(img, kernel):
    return convolution(img, kernel)

def gauscolor(img, kernel=guassfiltr):
    img = Spatialcolor(img, kernel)
    rows, cols, val = img.shape
    
    img = cv2.resize(img,(int(rows / 2), int(cols / 2)))
    return img
def convolution(img, kernel):
    
    outImg = np.zeros(img.shape)
    kernelvalsum = np.sum(kernel)

    if kernelvalsum == 0:
        kernelvalsum = 1

    width = kernel.shape[0]
    if width == 3:
        offset = 1
        start = -1
        end = 2

    if width == 5:
        offset = 2
        start = -2
        end = 3

    if width == 7:
        offset = 3
        start = -3
        end = 4

    for m in range(3):
        imgvaltemp = img[:, :, m]
        for i in range(1, imgvaltemp.shape[0] - offset):
            for j in range(1, imgvaltemp.shape[1] - offset):
                temp = 0
                for k in range(start, end):
                    for l in range(start, end):
                        temp += kernel[k][l] * imgvaltemp[i + k][j + l]

                outImg[i][j][m] = temp / kernelvalsum

    return np.(outImg)


def gaussianvalpyramidcolor(img, number):
    G = img.copy()

    gpyr = [G]
    for i in range(number):
        G = gauscolor(G)
        

        gpyr.append(G)

    return gpyr
gpcolorval1 = gaussianvalpyramidcolor(pikachu, number=4)
gpcolorval2 = gaussianvalpyramidcolor(mewoth, number=4)

def laplacianvalpyramidcolor(gp):
    length = len(gp)

    lp = []

    for i in range(length - 1, 0, -1):
        temp = gp[i]
        rows, cols, val = temp.shape
        up = cv2.resize(gp[i],(2 * rows, 2 * cols))
        
        up = cv2.resize(up, (gp[i - 1].shape[0], gp[i - 1].shape[1]))
        lap = cv2.subtract(gp[i - 1], up)
        lp.append(lap)
        
    return lp

lpcolorval1 = laplacianvalpyramidcolor(gpcolorval1)
lpcolorval2 = laplacianvalpyramidcolor(gpcolorval2)

def padcolor(img, padvalsize):
    row, col = img.shape
    finalvalrow, finalcolorol = row + padvalsize, col + padvalsize

    finalvalimg = np.ones([finalvalrow, finalcolorol])

    for i in range(padvalsize, row):
        for j in range(padvalsize, col):
            finalvalimg[i][j] = img[i - padvalsize][j - padvalsize]
    return finalvalimg


def pyrupvalupsamplecolor(img, ratio=2):
    row, col, val = img.shape
    finalvalrow, finalcolorol = 2 * row, col * 2

    finalvalimg = np.zeros([finalvalrow, finalcolorol, 3])

    for m in range(3):
        for i in range(0, finalvalrow, 2):
            for j in range(0, finalcolorol, 2):
                finalvalimg[i][j][m] = img[int(i / 2)][int(j / 2)][m]
    return finalvalimg


def pyrupcolor(lsval, LS):
    lsval = pyrupvalupsamplecolor(lsval)
    
    lsval = cv2.resize(lsval, (LS.shape[0], LS.shape[1]))
    lsval = convolution(lsval, 4 * guassfiltr)
    lsval = np.uint8(lsval)
    return lsval


def hybridcolor(Lp1, Lp2):
    LS = []

    for la, lb in zip(Lp1, Lp2):
        LS.append((la +  lb))

    lsval = LS[0]
    for i in range(1, len(LS)):
        lsval = pyrupcolor(lsval, LS[i])
        lsval = cv2.add(lsval, LS[i])

    return lsval, LS
image, out = hybridcolor(lpcolorval1, lpcolorval2)
cv2.imwrite('colorhybrid.png',image)
cv2.imshow("colorHybrid",image)    
cv2.waitKey(0)
cv2.destroyAllWindows() 

