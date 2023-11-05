from sys import argv
from time import process_time

from tools import *

# Timing
start = time()
cpu_start = process_time()

# Read
path = argv[1]
mode = argv[2]
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_UNCHANGED keeps the alpha channel, instead of removing it
height, width, channels = img.shape


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


def blur(init):
	arr = Kernel(np.full((init, init), 1))

	im = convolve(img, arr)
	cv2.imwrite(f"blured_{path.split('/')[-1]}", im)


def gaussian_blur(stdev):
	a = Gaussian(stdev)

	# tools.vis(a.matrix)
	im = convolve(img, a)
	cv2.imwrite(f"blured_{path.split('/')[-1]}", im)


def edges(init):
	r = np.array([[-(init / 2), -init, -(init / 2)], [0, 0, 0], [init / 2, init, init / 2]])
	hor = Kernel(r)
	ver = Kernel(r.transpose())

	# tools.vis(r)
	# tools.vis(r.transpose())
	im = np.sqrt(convolve(img, hor) ** 2 + convolve(img, ver) ** 2)
	cv2.imwrite(f"testFINAL.png", im)


def grayscale():
	im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	cv2.imwrite(f"grayscaled.png", im)


def sharpen(init):
	r = np.array([[0, -(init / 4), 0], [-(init / 4), init + 1, -(init / 4)], [0, -(init / 4), 0]])
	a = Kernel(r)

	im = convolve(img, a)
	cv2.imwrite(f"sharpened.png", im)


def main():
	if mode == "--grayscale" or mode == "--gray":
		grayscale()
	elif mode == "--inverse":
		inverse()
	elif mode == "--edges":
		parameter = 0.5
		if len(argv) == 4:
			parameter = int(argv[3])
		edges(parameter)
	elif mode == "--boxblur" or mode == "--blur":
		parameter = 3
		if len(argv) == 4:
			parameter = int(argv[3])
		blur(parameter)
	elif mode == "--gaussianblur" or mode == "--gaussian":
		parameter = 1
		if len(argv) == 4:
			parameter = int(argv[3])
		gaussian_blur(parameter)
	elif mode == "--sharpen":
		parameter = 4
		if len(argv) == 4:
			parameter = int(argv[3])
		sharpen(parameter)


main()

end = time()
cpu_end = process_time()

print(f"Wall-Clock: {end - start}\nCPU: {cpu_end - cpu_start}")
