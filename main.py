import cv2
from sys import argv
from time import time, process_time

from kernels import *
import tools

# Timing
start = time()
cpu_start = process_time()

# Read
path = argv[1]
mode = argv[2]
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_UNCHANGED keeps the alpha channel, instead of removing it
height, width, channels = img.shape


def convolve(image, kernel: Kernel):
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


def blur():
	arr = Kernel(np.full((3, 3), 1))

	im = convolve(img, arr)
	cv2.imwrite(f"blured_{path.split('/')[-1]}", im)


def gaussian_blur(stdev):
	a = Gaussian(stdev)
	# tools.vis(a.matrix)

	im = convolve(img, a)
	cv2.imwrite(f"blured_{path.split('/')[-1]}", im)


def edges():
	r = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])
	hor = Kernel(r)
	ver = Kernel(r.transpose())

	# im = np.sqrt(convolve(img, hor) ** 2 + convolve(img, ver) ** 2)
	# cv2.imwrite(f"testFINAL.png", im)
	tools.vis(r.transpose())


def grayscale():
	im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	cv2.imwrite(f"grayscaled.png", im)


def main():
	if mode == "--grayscale" or mode == "--gray":
		grayscale()
	elif mode == "--inverse":
		inverse()
	elif mode == "--edges":
		edges()
	elif mode == "--boxblur" or mode == "--blur":
		blur()
	elif mode == "--gaussianblur" or mode == "--gaussian":
		parameter = 1
		if len(argv) == 4:
			parameter = int(argv[3])
		gaussian_blur(parameter)


main()

end = time()
cpu_end = process_time()

print(f"Wall-Clock: {end - start}\nCPU: {cpu_end - cpu_start}")
