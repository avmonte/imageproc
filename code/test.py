import cv2
import numpy as np

from tools import vis
from helper import skeletonize_image

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


def create_angle_kernel(size, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    kernel = np.zeros((size, size), dtype=np.uint8)

    # Assuming the line passes through the center of the kernel
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            # Check if the point (x, y) lies close to the line with the given angle
            if abs(y * np.cos(angle_radians) - x * np.sin(angle_radians)) < 1:
                kernel[i, j] = 1
    return kernel

# Example usage
kernel = create_angle_kernel(5, 45)  # 5x5 kernel with a 45-degree line
print(kernel)

kernel = create_angle_kernel(5, 90)  # 5x5 kernel with a 90-degree line
vis(kernel)

im = cv2.imread('../examples/16.jpg')
kernel = Kernel(3).generate(16)
# im to binary
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]

im = cv2.erode(im, kernel, iterations=1)
im = cv2.filter2D(im, -1, kernel)
im = cv2.filter2D(im, -1, kernel)

for i in range(5):
    im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]
    im = cv2.dilate(im, kernel, iterations=1)
    im = cv2.erode(im, kernel, iterations=1)
    im = cv2.filter2D(im, -1, kernel)
    im = cv2.filter2D(im, -1, kernel)

im = cv2.filter2D(im, -1, kernel)
im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]

im = cv2.bitwise_not(im)
# im = skeletonize_image(im)

# print('skeletonized')
im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('../examples/gevorgsblurMEGA.jpg', im)
