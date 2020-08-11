import numpy as np
import cv2
import matplotlib.pyplot as plt

coins = cv2.imread('coins.png',cv2.IMREAD_UNCHANGED)
coins = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
print(coins.shape)
cv2.imshow("Original image", coins)
cv2.waitKey(0)
cv2.destroyAllWindows()
coinarr = np.array(coins)
for  i in range(coinarr.shape[0]):
    for j in range(coinarr.shape[1]):
        
        if coinarr[i][j] > 128:
            coinarr[i][j] = 255
        else:
            coinarr[i][j] = 0

cv2.imshow("Original image", coinarr)
cv2.waitKey(0)
cv2.destroyAllWindows()    

kernel = np.ones((5,5), np.uint8)   

greycoin = coinarr
greycoin = np.pad(greycoin, (2, 2), 'constant', constant_values=(0, 0))

     
def conv(greycoin,kernel):
    erode = np.full((greycoin.shape[0],greycoin.shape[1]),255)
    
    for i in range(2,greycoin.shape[0]-3):
        for j in range(2,greycoin.shape[1]-3):
            
            val = np.sum(np.multiply(greycoin[i-2:i+3,j-2:j+3],kernel))
            
            if val ==0:
                erode[i][j] =0
            else:
                
                erode[i][j] = 255
    
    erode = erode.astype(np.uint8)            
    return erode
                     
erode = conv(greycoin,kernel)  
cv2.imshow("Eroded image", erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
kernel = np.ones((12,12))
greycoin = 255 - greycoin


img_erosion = cv2.erode(greycoin, kernel, iterations=1)
greycoin = 255 - greycoin
img_erosion = 255 - img_erosion
cv2.imshow("Eroded image", img_erosion)
cv2.imwrite('coin-erode.png',img_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
cont_img = img_erosion.copy()
contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 2000 or area > 4000:
        continue
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(cont_img, ellipse, (0,255,0), 2)

outline = cv2.Canny(img_erosion, 30, 150)
cv2.imshow("The edges", outline)
cv2.waitKey(0)
(cnts, _) = cv2.findContours(outline, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_erosion, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Result", img_erosion)
cv2.waitKey(0)
print("Total %i coins" % len(cnts))