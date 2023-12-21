import cv2
import numpy as np

# Load the image
image = cv2.imread('blurred_edition.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to get a binary image
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define a kernel for morphological operations
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))

# Apply a combination of dilation and erosion to consolidate the lines
dilated = cv2.dilate(binary, rect_kernel, iterations=1)
eroded = cv2.erode(dilated, rect_kernel, iterations=1)

# Find contours of the lines
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from top to bottom based on their y-coordinate
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

# Draw bounding boxes around each line of text
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with identified lines
cv2.imwrite('lines.jpg', image)