import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
Main function
"""
def main():
    # Load the images
    imageLeft = cv2.imread("images/office1.jpg")
    imageMiddle = cv2.imread("images/office2.jpg")
    imageRight = cv2.imread("images/office3.jpg")

    
    imageLeft = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)
    imageMiddle = cv2.cvtColor(imageMiddle, cv2.COLOR_BGR2GRAY)
    imageRight = cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)

    
    plt.imshow(imageLeft, cmap="gray")
    plt.axis("off")
    plt.title("Left image")
    plt.show()

    plt.imshow(imageMiddle, cmap="gray")
    plt.axis("off")
    plt.title("Middle image")
    plt.show()

    plt.imshow(imageRight, cmap="gray")
    plt.axis("off")
    plt.title("Right image")
    plt.show()

    orb = cv2.ORB_create()

    
    kpLeft, desLeft = orb.detectAndCompute(imageLeft, None)
    kpMiddle, desMiddle = orb.detectAndCompute(imageMiddle, None)
    kpRight, desRight = orb.detectAndCompute(imageRight, None)

    
    BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
    matchesLeftMiddle = BFMatcher.match(desLeft,desMiddle)

    
    matchesRightMiddle = BFMatcher.match(desRight,desMiddle)
    
    srcPtsLeftMiddle = np.float32([kpLeft[matchesLeftMiddle[match].queryIdx].pt for match in range(0, 50)])
    desPtsLeftMiddle = np.float32([kpMiddle[matchesLeftMiddle[match].trainIdx].pt for match in range(0, 50)])

    srcPtsRightMiddle = np.float32([kpRight[matchesRightMiddle[match].queryIdx].pt for match in range(0, 50)])
    desPtsRightMiddle = np.float32([kpMiddle[matchesRightMiddle[match].trainIdx].pt for match in range(0, 50)])

    
    height, width = imageMiddle.shape

    
    mLeft, maskLeft = cv2.findHomography(srcPtsLeftMiddle, desPtsLeftMiddle, cv2.RANSAC)
    warpLeftMiddle = cv2.warpPerspective(imageLeft, mLeft, (width, height))
    mergedLeftMiddle = cv2.bitwise_or(warpLeftMiddle, imageMiddle)
    mRight, maskRight = cv2.findHomography(srcPtsRightMiddle, desPtsRightMiddle, cv2.RANSAC)
    warpRightMiddle = cv2.warpPerspective(imageRight, mRight, (width, height))
    mergedFinal = cv2.bitwise_or(warpRightMiddle, mergedLeftMiddle)

    # Show the final image
    cv2.imwrite('images/output1.jpg', mergedFinal)
    plt.imshow(mergedFinal, cmap="gray")
    plt.axis("off")
    plt.title("Result Image")
    plt.show()

if __name__ == '__main__':
   main()