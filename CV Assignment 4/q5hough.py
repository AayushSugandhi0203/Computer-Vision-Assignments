import numpy as np
import imageio
import math

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
