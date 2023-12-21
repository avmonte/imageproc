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
                    arr[i, j] = gaussian(x, y)

        print(np.sum(arr, dtype=np.float64), n)  # for debug
        return arr / np.sum(arr, dtype=np.float64)  # normalize


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

    cv2.imwrite('red_removed.jpg', image)
    return image


def convert_to_binary_and_save(image):  # EXAMPLE: T7

    # Sharpen and Blur the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)  # cv2.filter2D(image, -1, kernel) can be used for a faster computation

    blurred = cv2.GaussianBlur(sharpened, (5, 5), 5)

    # To grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # # Apply automatic contrast
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # contrast_enhanced = clahe.apply(gray)

    # To binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite('binary_edition.jpg', binary)
    return binary


def skeletonize_image(binary_image):
    """
    Skeletonizes the given binary image using morphological transformations.

    Parameters:
    binary_image (numpy.ndarray): A binary image (black and white only).

    Returns:
    numpy.ndarray: The skeletonized version of the input image.
    """

    # Initialize the skeleton
    skeleton = np.zeros(binary_image.shape, np.uint8)

    # Get a cross-shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Erode the image
        eroded = cv2.erode(binary_image, element)

        # Dilate the eroded image
        temp = cv2.dilate(eroded, element)

        # Subtract the dilated image from the original image to get the edge (skeleton)
        temp = cv2.subtract(binary_image, temp)

        # Or the skeleton with the temp image
        skeleton = cv2.bitwise_or(skeleton, temp)

        # Update the image for next iteration
        binary_image = eroded.copy()

        # If the image has been completely eroded away, break out of the loop
        if cv2.countNonZero(binary_image) == 0:
            break

    return skeleton


def find_most_frequent(values, bin_width):
    # Create a histogram with the given bin width
    bins = np.arange(min(values), max(values) + bin_width, bin_width)
    histogram, bins = np.histogram(values, bins=bins)

    # Find the index of the max frequency bin
    max_freq_idx = np.argmax(histogram)

    # Find the most frequent bin range
    most_frequent_range = (bins[max_freq_idx], bins[max_freq_idx + 1])

    # Calculate the midpoint of the most frequent bin range as the representative value
    most_frequent_value = np.mean(most_frequent_range)

    return most_frequent_value


def iterate(image):
    ## Prism Gaussian Blur
    global angle

    if angle is None:
        blurred = cv2.GaussianBlur(image, (5, 5), 5)
        cv2.imwrite('blurred_edition.jpg', blurred)  # NOTE
    else:
        kernel = Kernel(3).generate(angle)
        blurred = cv2.filter2D(image, -1, kernel)

    ## Edge detection
    thresh = cv2.Canny(blurred, 50, 150, apertureSize=3)

    thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=10)
    thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=10)
    # thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=5)
    # thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=5)

    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 10)

    thresh = skeletonize_image(thresh)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 10)

    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)

    cv2.imwrite('thresh.jpg', thresh)
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

        cv2.imwrite('lines2.jpg', cimg)

        # average of theta values from Hough Transform
        print(lines[:, 0][:, 1])

        theta_radians = find_most_frequent(lines[:, 0][:, 1], 0.05)

        # Convert theta to degrees
        theta_degrees = np.degrees(theta_radians)

        # Calculate the angle of the line with the x-axis
        angle = 90 - theta_degrees

        print("Average Theta (degrees):", round(angle))
        if angle < 0:
            angle = 180 + angle
        print("Error:", abs(round(actual_angle - angle, 2)))


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
    filename = '114.jpg'
    img = cv2.imread(filename)
    actual_angle = int(filename.split('.')[0])

    # img = remove_red_marks(img)
    img = convert_to_binary_and_save(img)

    for i in range(2):
        iterate(img)
    if angle is not None:
        rotate(img)

    # save image
    cv2.imwrite(f'{actual_angle}p.jpg', img)

main()
