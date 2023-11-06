import cv2

# Read the pixel art image using OpenCV
input_image = cv2.imread("../processed/kernel_visualization/gaussian100.png")

# Resize the image to 128x128 pixels without interpolation (no blurring)
output_image = cv2.resize(input_image, (128, 128), interpolation=cv2.INTER_NEAREST)

# Save the resized image
cv2.imwrite("../processed/kernel_visualization/e100.png", output_image)

