
# from tools import vis
from helper import find_most_frequent, skeletonize_image, convert_to_binary_and_save, remove_red_marks
import cv2
import numpy as np

"""
PLAN

Gaussian Blur --> Edge -> Hough -> Angle
Prism Gaussian Blur --> Edge -> Hough -> Angle 
Rotate

"""

angle = None


class Kernel:
    def __init__(self, stdev):
        self.stdev = stdev

    def generate(self, alpha):
        threshold = 0.97
        alpha_rad = np.radians(alpha)  # Convert alpha to radians
        a = np.cos(alpha_rad)
        b = np.sin(alpha_rad)

        amp = 1 / 100

        gaussian = lambda x, y: amp * np.exp(-((a * x + b * y) ** 2) / (2 * self.stdev ** 2))
        n = 7
        arr = np.zeros(shape=(n, n, 1), dtype=np.float64)

        while np.sum(arr, dtype=np.float64) < threshold:
            n += 2
            arr = np.zeros(shape=(n, n, 1), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    x, y = i - (n // 2), j - (n // 2)
                    m = gaussian(x, y)
                    # print(m)
                    arr[i, j] = m

        # print(np.sum(arr, dtype=np.float64), n)  # for debug
        return arr / np.sum(arr, dtype=np.float64)  # normalize


def iterate(image):
    ## Prism Gaussian Blur
    global angle

    if angle is None:
        blurred = cv2.GaussianBlur(image, (5, 5), 5)
        cv2.imwrite('blurred_edition.jpg', blurred)  # NOTE
        print('Blurred')
    else:
        print(angle)
        kernel = Kernel(3).generate(angle)
        # vis(kernel)  # for debug
        blurred = cv2.filter2D(image, -1, kernel)

    ## Edge detection
    thresh = cv2.Canny(blurred, 50, 150, apertureSize=3)

    for i in range(6, 20, 2):
        thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=i)
        thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=i)
        # thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=5)
        # thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=5)

        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)
        if angle is None:
            thresh = cv2.GaussianBlur(thresh, (5, 5), 5)
        else:
            thresh = cv2.filter2D(thresh, -1, kernel)

        thresh = skeletonize_image(thresh)
        if angle is None:
            thresh = cv2.GaussianBlur(thresh, (5, 5), 5)
        else:
            thresh = cv2.filter2D(thresh, -1, kernel)

        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)

        cv2.imwrite('thresh.jpg', thresh)
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

        cv2.imwrite('lines2.jpg', cimg)

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
    filename = '../examples/114.jpg'
    img = cv2.imread(filename)
    actual_angle = int((filename.split('.')[-2]).split('/')[-1])

    # img = remove_red_marks(img)
    img = convert_to_binary_and_save(img)

    for i in range(2):
        iterate(img)
        # print(angle) # for debug

    if angle is not None:
        img = rotate(cv2.imread(filename))

    # save image
    cv2.imwrite(f'../examples/{actual_angle}p.jpg', img)

main()
print('Done')
