import cv2
import numpy as np
from sys import argv
from time import time, process_time

# Timing
start = time()
cpu_start = process_time()

# Read
path = argv[1]
mode = argv[2]
img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_UNCHANGED keeps the alpha channel, instead of removing it
height, width, channels = img.shape


class Kernel:

	def __init__(self, matrix):
		self.matrix = matrix
		self.size = matrix.shape[0] // 2
		self.coef = 1 / np.sum(matrix) if np.sum(matrix) != 0 else 1

	#check the two props


def convolve(image, kernel):
	h, w, c = image.shape
	final = np.zeros(shape=(h, w, c), dtype=np.int16)
	s = kernel.size

	for i in range(h):
		for j in range(w):
			total = np.array([0] * c)

			for k in range(i - s, i + s + 1):
				for m in range(j - s, j + s + 1):
					total += np.round(image[np.clip(k, 0, h-1), np.clip(m, 0, w-1)] * kernel.matrix[s + k - i, s + m - j]).astype(int)

			final[i, j] = abs(total * kernel.coef)

	return final


def inverse():
	channels_list = cv2.split(img)
	b, g, r = (channels_list[i] for i in range(3))

	# average color array = (minimal color array + maximal color array) / 2
	avg_color = (np.array([np.min(b), np.min(g), np.min(r)]) + np.array([np.max(b), np.max(g), np.max(r)])) / 2

	for i in range(height):
		for j in range(width):
			for k in range(3):
				''' new = average + difference with the average
					= average + (average - original)
					= 2 * average - original '''
				img[i, j][k] = 2 * avg_color[k] - img[i, j][k]

	# Save
	cv2.imwrite(f"inverse_{path.split('/')[-1]}", img)


def blur(kernel=(lambda x, y: 1 / 25)):
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

			summ = [0] * channels
			for k in range(i - 2, i + 3):
				for m in range(j - 2, j + 3):
					if 0 <= k < height and 0 <= m < width:
						summ += img[k, m] * kernel(i - k, j - m)

			blured_img[i, j] = summ

	# Save
	cv2.imwrite(f"blured_{path.split('/')[-1]}", blured_img)


def gaussian_blur(stdev):
	gaussian = lambda x, y: (np.exp(-(x ** 2 + y ** 2) / (2 * (stdev ** 2) )) / (2 * np.pi * (stdev ** 2)))
	blur(gaussian)


def edges_test():
	r = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])
	hor = Kernel(r)
	ver = Kernel(r.transpose())

	im = np.sqrt(convolve(img, hor) ** 2 + convolve(img, ver) ** 2)
	cv2.imwrite(f"testFINAL.png", im)


def edges():
	edges_curr_img = np.zeros(shape=(height, width, channels), dtype=np.int16)
	edges_img = None

	kernel = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])

	for n in range(4):

		for i in range(height):
			for j in range(width):

				summ = [0] * channels
				for k in range(i - 1, i + 2):
					for m in range(j - 1, j + 2):
						if 0 <= k < height and 0 <= m < width:
							summ += img[k, m] * kernel[i - k, j - m]

				edges_curr_img[i, j] = summ

		if n == 0:  # To avoid unnecessary computations
			edges_img = edges_curr_img
		else:
			edges_img = np.sqrt(edges_img ** 2 + edges_curr_img ** 2)

		kernel = np.rot90(kernel, n + 1)


	# Save
	cv2.imwrite(f"edges44_{path.split('/')[-1]}", edges_img)

def grayscale():
	im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	cv2.imwrite(f"grayscalled.png", im)

def main():
	if mode == "--inverse":
		inverse()
	if mode == "--grayscale":
		grayscale()
	elif mode == "--boxblur":
		blur()
	elif mode == "--gaussianblur":
		parameter = 1
		if len(argv) == 4:
			parameter = int(argv[3])
		gaussian_blur(parameter)
	elif mode == "--edges":
		edges_test()

main()


end = time()
cpu_end = process_time()

print(f"Wall-Clock: {end - start}\nCPU: {cpu_end - cpu_start}")