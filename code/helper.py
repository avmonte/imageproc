import cv2
import numpy as np


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

