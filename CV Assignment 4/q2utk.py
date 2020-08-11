import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
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


