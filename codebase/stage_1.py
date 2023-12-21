"""
STAGE 1

1. MANUALLY: eliminate the printed text and crop the handwriting
2. AUTOMATED: remove the instructor’s marks in red
3. AUTOMATED: sharpen, blur and convert the cropped image to binary format
"""

import cv2
import math
import numpy as np
from datetime import datetime
from time import time


def remove_red_marks(image):  # EXAMPLE: T5
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # NOTE: OpenCV stores images in BGR format rather than RGB
            r = int(image[i][j][2])
            g = int(image[i][j][1])
            b = int(image[i][j][0])

            # if red is the highest value and green and blue are about the same
            if r > g and r > b and (g - b) < 30:
                image[i][j] = [255, 255, 255]

    return image


def convert_to_binary_and_save(image):  # EXAMPLE: T7

    # Sharpen and Blur the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)  # cv2.filter2D(image, -1, kernel) can be used for a faster computation

    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)

    # To grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Apply automatic contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # To binary
    _, binary = cv2.threshold(contrast_enhanced, 127, 255, cv2.THRESH_BINARY)

    return binary


def apply_hough_transform(image):  # EXAMPLE: T17
    ## Edge detection
    # edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply Hough Transform
    lines = cv2.HoughLines(thresh, 1, np.pi / 180, 200)

    cimg = np.copy(image)  # EXAMPLE: T11
    # Draw the lines on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(cimg, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # average of theta values from Hough Transform
    theta_radians = np.mean(lines[:, 0][:, 1])

    # Convert theta to degrees
    theta_degrees = math.degrees(theta_radians)

    # Calculate the angle of the line with the x-axis
    angle = 90 - theta_degrees
    # TODO: save globally

    print("Average Theta (degrees):", angle, round(angle, 2))

    # Rotate the image if the angle < 10 degrees
    if abs(angle) < 10:
        # Get the dimensions of the image
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -round(angle, 2), 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(255, 255, 255))

    return image


def main():
    ## Read

    filename = 'MD.M108_p1.jpg'
    im = cv2.imread(filename)

    ## Image Manipulation

    # im = remove_red_marks(im)
    # im = convert_to_binary_and_save(im)
    # im = apply_hough_transform(im)

    ## Write

    # today = datetime.now().strftime("%d%m%y")
    today = str(round(time()))[-6:]  # consistent for 11 days after dec 21st 2023
    cv2.imwrite(f'CRS.TST.{today}.{filename.split(".")[0]}_bin.png', im)


main()
