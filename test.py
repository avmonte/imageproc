import cv2
import numpy as np
import math
# read image
img = cv2.imread('../test.png')

# apply horizontal blur i.e smth like [[1, 1, 1][3, 3, 3][1, 1, 1]] but normalized
img = cv2.blur(img, (3, 3))

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# save image
cv2.imwrite('test1.png', img)


"""
PLAN

Gaussian Blur -> Canny Edge Detection -> Hough Transform -> Find the angle approximation
Prism Gaussian Blur -> Canny Edge Detection -> Hough Transform -> Find the angle -> Rotate the image -> Crop the image

"""


def initial_approx(image):  # EXAMPLE: T17
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    ## Edge detection
    # edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
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

    print("Average Theta (degrees):", angle, round(angle, 2))

    # Rotate the image if the angle < 10 degrees
    if abs(angle) < 10:
        # Get the dimensions of the image
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -round(angle, 2), 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(255, 255, 255))

    return image


def prism_approx(image):
	## Prism Gaussian Blur







	## Edge detection
	# edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
	gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
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