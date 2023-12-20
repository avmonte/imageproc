"""
STAGE 1

1. MANUALLY: eliminate the printed text and crop the handwriting
2. AUTOMATED: remove the instructor’s marks in red
3. AUTOMATED: sharpen and convert the cropped image to binary format
"""

from tools import *
from kernels import *

import cv2
import numpy as np


def remove_red_marks(image_path):  #T5

    # Read the image
    image = cv2.imread(image_path)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = int(image[i][j][2])
            g = int(image[i][j][1])
            b = int(image[i][j][0])

            if r > g and r > b and (g - b) < 30:
                image[i][j] = [255, 255, 255]

    return image


def convert_to_binary_and_save(image, filename):  # T7

    # increase sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(image, -1, kernel)  # cv2.filter2D(image, -1, kernel) can be used for a faster computation

    # Convert to grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    # Apply automatic contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    # Convert to binary
    _, binary = cv2.threshold(contrast_enhanced, 127, 255, cv2.THRESH_BINARY)
    # Save image
    cv2.imwrite(filename, binary)


def find_lines(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite('S0.jpg', blurred)
    lines = cv2.HoughLinesP(blurred, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red lines

    return line_image


def apply_hough_transform(image_path):
    # Read the image
    img = cv2.imread(image_path)

    blurred = cv2.GaussianBlur(img, (5, 5), 5)

    # Edge detection
    # edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply Hough Transform
    lines = cv2.HoughLines(thresh, 1, np.pi / 180, 200)

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
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # Save or display the result
    cv2.imwrite('T12.jpg', img)
    # Optionally display the result in a window
    # cv2.imshow('Result Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# im = remove_red_marks('MD.M106_p2.jpg')
# convert_to_binary_and_save(im, 'T8.jpg')

# im = cv2.imread('T7.jpg')
# cv2.imwrite('T9.jpg', im)

apply_hough_transform('T7.jpg')



"""
Stage 1 – eliminate the printed text (Chapters 3, 4): 

Study the color distribution in each image using RGB and grayscale histograms. 
Suggest and test a method for automated removal of the printed test.
 If needed, suggest and test a method for automated removal of instructor’s marks 
 (normally appearing in red shades). Having the handwriting extracted, suggest a test a method 
 to crop it. Convert the cropped image to binary format and save it under a filename 
 CRS.TST.ddmmyy.Cxxx_p1_bin.png. Adjust the brightness / contrast before the conversion as needed
 (for example, auto-contrast, histogram match, etc.).
"""
