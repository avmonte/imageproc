
# from tools import vis
from helper import *
import cv2
import numpy as np

"""
PLAN

Gaussian Blur --> Edge -> Hough -> Angle
Prism Gaussian Blur --> Edge -> Hough -> Angle 
Rotate

"""

angle = None


def enhanced_approx(image):
    ## Prism Gaussian Blur
    global angle

    kernel = Kernel().generate_prism_gaussian(angle, 3)
    dilate = Kernel().generate_DE(angle)

    # # vis(kernel)  # for debug
    # blurred = cv2.filter2D(image, -1, kernel)

    ## Edge detection
    im = cv2.erode(image, kernel, iterations=1)
    im = cv2.filter2D(im, -1, kernel)
    im = cv2.filter2D(im, -1, kernel)

    cv2.imwrite('../examples/16stages/blurred_edition2.jpg', im)  # NOTE

    for i in range(20):
        im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]
        im = cv2.dilate(im, dilate, iterations=1)
        im = cv2.erode(im, dilate, iterations=1)
        im = cv2.filter2D(im, -1, kernel)
        im = cv2.filter2D(im, -1, kernel)

    im = cv2.filter2D(im, -1, kernel)
    im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]
    im = cv2.bitwise_not(im)
    im = skeletonize_image(im)
    im = cv2.dilate(im, dilate, iterations=1)
    im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite('../examples/16stages/thresh2.jpg', im)

    lines = cv2.HoughLines(im, 1, np.pi / 180, 200)

    cimg = np.copy(image)  # EXAMPLE: T11
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

        cv2.imwrite('../examples/16stages/lines2.jpg', cimg)

        # average of theta values from Hough Transform
        # print(lines[:, 0][:, 1])  # for debug

        theta_radians = find_most_frequent(lines[:, 0][:, 1], 0.05)

        # Convert theta to degrees
        theta_degrees = np.degrees(theta_radians)

        # Calculate the angle of the line with the x-axis
        angle = round(90 - theta_degrees, 3)

        print("Average Theta (degrees):", round(angle))
        if angle < 0:
            angle = 180 + angle
        print("Error:", abs(round(actual_angle - angle, 2)), '\n')


def initial_approx(image):
    ## Prism Gaussian Blur
    global angle

    blurred = cv2.GaussianBlur(image, (5, 5), 5)
    cv2.imwrite('../examples/16stages/blurred_edition.jpg', blurred)  # NOTE
    print('Blurred')


    ## Edge detection
    thresh = cv2.Canny(blurred, 50, 150, apertureSize=3)

    for i in range(6, 20, 2):
        thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=i)
        thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=i)
        # thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=5)
        # thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=5)

        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 5)

        thresh = skeletonize_image(thresh)

        thresh = cv2.GaussianBlur(thresh, (5, 5), 5)

        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)

        cv2.imwrite('../examples/16stages/thresh.jpg', thresh)
        # Apply Hough Transform
        lines = cv2.HoughLines(thresh, 1, np.pi / 180, 200)

        if lines is not None:
            if len(lines) in range(20, 40):
                break

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

        cv2.imwrite('../examples/16stages/lines2.jpg', cimg)

        # average of theta values from Hough Transform
        # print(lines[:, 0][:, 1])  # for debug

        theta_radians = find_most_frequent(lines[:, 0][:, 1], 0.05)

        # Convert theta to degrees
        theta_degrees = np.degrees(theta_radians)

        # Calculate the angle of the line with the x-axis
        angle = round(90 - theta_degrees, 3)

        print("Average Theta (degrees):", round(angle))
        if angle < 0:
            angle = 180 + angle
        print("Error:", abs(round(actual_angle - angle, 2)), '\n')


def rotate(image):
    global angle
    # Get the dimensions of the image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -round(angle, 2), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(255, 255, 255))

    return rotated_image


def main():
    global actual_angle
    # read image
    filename = '../examples/16.jpg'
    img = cv2.imread(filename)
    actual_angle = int((filename.split('.')[-2]).split('/')[-1])

    # img = remove_red_marks(img)
    img = convert_to_binary_and_save(img)

    initial_approx(img)
    enhanced_approx(img)

    if angle is not None:
        img = rotate(cv2.imread(filename))

    # save image
    cv2.imwrite(f'../examples/{actual_angle}p.jpg', img)

main()
print('Done')
