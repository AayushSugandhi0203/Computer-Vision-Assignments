import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
coins = cv2.imread('sih.png',cv2.IMREAD_UNCHANGED)
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
#################Q2
kernel = np.ones((3, 3), dtype=np.int64)
sel = kernel

utk=cv2.imread('UTK.PNG',0)
t=cv2.imread('T.PNG',0)
cv2.imshow('utk',utk)
cv2.waitKey(0)
print(utk.shape)
cv2.imshow('t',t)
cv2.waitKey(0)
print(t.shape)
img = utk
def apply_threshold(img, threshold=.5):

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] > threshold:
                img[x,y] = 255
            else:
                img[x,y] = 0

    return img

img1 = apply_threshold(utk)
cv2.imshow('Img',img1)
cv2.waitKey(0)
as_gray = img1
def crop_zero_values(img, inverted=False):
    
    width, height = img.shape
    xmin, xmax = 0, width
    ymin, ymax = 0, height
    ones_x, ones_y = np.where(img == 1) if not inverted else np.where(img == 0)
    if ones_x.size > 0:
        xmin, xmax = min(ones_x), max(ones_x)
    if ones_y.size > 0:
        ymin, ymax = min(ones_y), max(ones_y)
    return img[xmin:xmax+1, ymin:ymax+1]

img = crop_zero_values(img)
cv2.imshow('Img',img)
cv2.waitKey(0)
def apply_erosion(neighbors, as_gray):
    
    if not as_gray:
        if max(neighbors.ravel()) == 1:
            
                return 1
        return 0
    return min(neighbors.ravel())

def apply_dilation(neighbors, as_gray):
    
    if not as_gray:
        if max(neighbors.ravel()) == 0:
            return 0
        return 1
    return max(neighbors.ravel())

def add_padding(img, radius):
    width, height = img.shape
    pad_img_shape = (width + radius - 1, height + radius - 1)
    pad_img = np.zeros(pad_img_shape).astype(np.float64)
    pad_img[radius-2:(width + radius-2), radius-2:(height + radius-2)] = img
    return pad_img

img = add_padding(img,4)
cv2.imshow('Img',img)
cv2.waitKey(0)
def process_pixel(i, j, operation, as_gray, img):
    radius = kernel.shape[0]
    neighbors = img[i:i+radius, j:j+radius]
    if as_gray:
        neighbors = np.delete(neighbors.flatten(), radius+1)
    return operation(neighbors, as_gray)
output_image = cv2.morphologyEx(utk, cv2.MORPH_HITMISS, t)
def apply_filter(operation, img, as_gray, n_iterations, sel):
    
    global kernel
    kernel = sel
    
    width, height = img.shape
    prod = product(range(width), range(height))
    img_result = np.zeros_like(img)
    radius = kernel.shape[0]
    pad_img = add_padding(img, radius)
    if n_iterations >= 1:
        for i, j in prod:
            if operation == 'er' and pad_img[i, j] == 1:
                img_result[i, j] = process_pixel(i, j, operation, as_gray, pad_img)
            else:
                img_result[i, j] = process_pixel(i, j, operation, as_gray, pad_img)
        return apply_filter(operation, img_result, as_gray, n_iterations-1, sel)
    return img

def erosion(img, as_gray, n_iterations, sel):
    
    return apply_filter(apply_erosion, img, as_gray, n_iterations, sel)



def dilation(img, as_gray, n_iterations, sel):
    return apply_filter(apply_dilation, img, as_gray, n_iterations, sel)


def opening(img, as_gray, n_iterations, sel):
    
    eroded = erosion(img, as_gray, n_iterations, sel)
    dilated = dilation(eroded, as_gray, n_iterations, sel)
    return dilated


def closing(img, as_gray, n_iterations, sel):
    
    dilated = dilation(img, as_gray, n_iterations, sel)
    return erosion(dilated, as_gray, n_iterations, sel)
cv2.imshow("Hit or Miss", t)
cv2.waitKey(0)
cv2.destroyAllWindows()
input_image = utk

rate = 50
kernel = (kernel + 1) * 127
kernel = np.uint8(kernel)

kernel = cv2.resize(kernel, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)

cv2.moveWindow("kernel", 0, 0)

input_image = cv2.resize(input_image, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)

cv2.moveWindow("Original", 0, 200)

output_image = cv2.resize(output_image, None , fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)

cv2.moveWindow("Hit or Miss", 500, 200)

########Q3
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


################Q4
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


            
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            

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

#############Q5

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
   
    
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    plt.show()
import cv2
img  = cv2.imread('IITJammu_North_1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img',img)
print(img.shape)
cv2.waitKey()
edges = cv2.Canny(img, 50, 200)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, minLineLength=10, maxLineGap=250,threshold=128)

for line in lines:

    x1, y1, x2, y2 = line[0]

    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)




accumulator, thetas, rhos = hough_line(img)
print(accumulator,thetas,rhos)
im = show_hough_line(img, accumulator,thetas,rhos)
cv2.imshow("Result Image", img)
cv2.waitKey()
