import cv2
import numpy as np
import sys

# Read
filename = sys.argv[1]
mode = sys.argv[2]
image = cv2.imread(filename, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_UNCHANGED keeps the alpha channel, instead of removing it
height, width, channels = image.shape


def inverse():
	channels_list = cv2.split(image)
	b, g, r = (channels_list[i] for i in range(3))

	# average color array = (minimal color array + maximal color array) / 2
	avg_color = (np.array([np.min(b), np.min(g), np.min(r)]) + np.array([np.max(b), np.max(g), np.max(r)])) / 2

	for i in range(height):
		for j in range(width):
			for k in range(3):
				''' new = average + difference with the average
					= average + (average - original)
					= 2 * average - original '''
				image[i, j][k] = 2 * avg_color[k] - image[i, j][k]

	# Save
	cv2.imwrite(f"inverse_{filename}", image)


if mode == "--inverse":
	inverse()

