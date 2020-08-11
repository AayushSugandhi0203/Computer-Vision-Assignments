import numpy as np
import cv2
import matplotlib.pyplot as plt

circlelines= cv2.imread('circles_lines.png',cv2.IMREAD_UNCHANGED)
circlelines = cv2.cvtColor(circlelines, cv2.COLOR_BGR2GRAY)
print(circlelines.shape)
cv2.imshow("Original image", circlelines)
cv2.waitKey(0)
cv2.destroyAllWindows()
circlelinesarr = np.array(circlelines)

for  i in range(circlelinesarr.shape[0]):
    for j in range(circlelinesarr.shape[1]):
        
        if circlelinesarr[i][j] > 128:
            circlelinesarr[i][j] = 255
        else:
            circlelinesarr[i][j] = 0

   

from skimage import draw
arr = np.zeros((20, 20))
stroke = 3
# Create an outer and inner circle. Then subtract the inner from the outer.
radius = 6
inner_radius = radius - (stroke // 2) + (stroke % 2) - 1 
outer_radius = radius + ((stroke + 1) // 2)
ri, ci = draw.circle(10, 10, radius=inner_radius, shape=arr.shape)
ro, co = draw.circle(10, 10, radius=outer_radius, shape=arr.shape)
arr[ro, co] = 1
arr[ri, ci] = 0
radius = 5
a = np.zeros((12, 12)).astype('uint8')
cx, cy = 6, 6 # The center of circle
y, x = np.ogrid[-radius: radius, -radius: radius]
index = x**2 + y**2 <= radius**2
a[cy-radius:cy+radius, cx-radius:cx+radius][index] = 255

kernel = a  

greyirclelines = circlelinesarr
greyirclelines = np.pad(greyirclelines, (2, 2), 'constant', constant_values=(0, 0))

     
def converode(greyirclelines,kernel):
    erode = np.full((greyirclelines.shape[0],greyirclelines.shape[1]),255)
    
    for i in range(2,greyirclelines.shape[0]-3):
        for j in range(2,greyirclelines.shape[1]-3):
            
            val = np.sum(np.multiply(greyirclelines[i-2:i+3,j-2:j+3],kernel))
            
            if val ==0:
                erode[i][j] =0
            else:
                
                erode[i][j] = 255
    
    erode = erode.astype(np.uint8)            
    return erode
def convdialate(greyirclelines,kernel):
    erode = np.full((greyirclelines.shape[0],greyirclelines.shape[1]),255)
    
    for i in range(2,greyirclelines.shape[0]-3):
        for j in range(2,greyirclelines.shape[1]-3):
            
            val = np.sum(np.multiply(greyirclelines[i-2:i+3,j-2:j+3],kernel))
            
            if val >=1:
                erode[i][j] =0
            else:
                
                erode[i][j] = 255
    
    erode = erode.astype(np.uint8)            
    return erode
                     


img_erosion = cv2.erode(greyirclelines, kernel, iterations=1)



print(kernel)
opening = cv2.morphologyEx(greyirclelines, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening image", opening)
cv2.imwrite('opening.png',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
v2 = greyirclelines - opening

image = cv2.imread('circles_lines.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(thresh, [c], -1, (255,255,255), -1)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)


cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.drawContours(image, [c], -1, (36,255,12), 3)


cv2.imshow('opening', opening)
cv2.waitKey()
kernel = np.zeros((8,8), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
kernel = cv2.line(kernel,(0,0),(7,7),(255,255,255),5)
kernel2 = cv2.line(kernel,(7,0),(0,7),(255,255,255),5)

#img = converode(greyirclelines,kernel)
opening = cv2.morphologyEx(greyirclelines, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(greyirclelines, cv2.MORPH_OPEN, kernel2)
opening = opening + opening2



img = greyirclelines
#converted = convert_hls(img)

lower = np.uint8([0, 200, 0])
upper = np.uint8([255, 255, 255])
white_mask = cv2.inRange(image, lower, upper)
# yellow color mask
lower = np.uint8([10, 0,   100])
upper = np.uint8([40, 255, 255])
yellow_mask = cv2.inRange(image, lower, upper)
# combine the mask
mask = cv2.bitwise_or(white_mask, yellow_mask)
result = img.copy()
cv2.imshow("mask",mask)
cv2.waitKey() 
height,width = mask.shape
skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
temp_nonzero = np.count_nonzero(mask)
while(np.count_nonzero(mask) != 0 ):
    eroded = cv2.erode(mask,kernel)
      
    temp = cv2.dilate(eroded,kernel)
    
    temp = cv2.subtract(mask,temp)
    skel = cv2.bitwise_or(skel,temp)
    mask = eroded.copy()

cv2.imshow("skel",skel)

cv2.waitKey()