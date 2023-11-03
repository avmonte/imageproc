import cv2
import numpy as np
import sys

# Read
path = sys.argv[1]
mode = sys.argv[2]
image = cv2.imread(path, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_UNCHANGED keeps the alpha channel, instead of removing it
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
	cv2.imwrite(f"inverse_{path.split('/')[-1]}", image)


def blur():
	"""
	Our kernel values are uniform => no need for a variable to store those in.
	Just as we do not rotate it 180 degrees since per mathematical definition, uniform => symmetric.

	In each step of the loop we are going to calculate the convolution of kernel and 3x3 region around (i, j)-th pixel
	i.e.

		1/9 * image[i-1, j-1] + 1/9 * image[i-1, j] + ... + 1/9 * image[i, j] + ... + 1/9 * image[i+1, j+1]
	<=>
		1/9 * (image[i-1, j-1] + ... + image[i, j] + ... + image[i+1, j+1]),

	which is implemented with nested loops below
	"""

	blured_img = np.zeros(shape=(height, width, channels), dtype=np.int16)

	for i in range(height):
		for j in range(width):

			summ = [0, 0, 0]
			count = 0
			for k in range(i - 1, i + 2):
				for m in range(j - 1, j + 2):
					if 0 <= k < height and 0 <= m < width:
						summ += image[k, m]
						count += 1

			blured_img[i, j] = summ / count  ##

	# Save
	cv2.imwrite(f"blured_{path.split('/')[-1]}", blured_img)


if mode == "--inverse":
	inverse()
elif mode == "--blur":
	blur()
