from sys import argv
from time import process_time

from tools import *

# Timing
start = time()
cpu_start = process_time()

# Read
path = argv[1]
mode = argv[2]
try:
	parameter = argv[3]
except IndexError:
	parameter = None

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_UNCHANGED keeps the alpha channel, instead of removing it
try:
	height, width, channels = img.shape
except ValueError:
	height, width, channels = img.shape + tuple([1])


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
	# vis(arr.matrix)

	im = convolve(img, arr)
	cv2.imwrite(f"blured_{path.split('/')[-1]}", im)


def gaussian_blur(stdev):
	im = convolve(img, Gaussian(stdev))
	cv2.imwrite(f"blured_{path.split('/')[-1]}", im)


def edges(init):
	# vis(EdgeKernel(init).matrix)
	# vis(EdgeKernel(init, False).matrix)

	im = np.sqrt(convolve(img, EdgeKernel(init)) ** 2 + convolve(img, EdgeKernel(init, False)) ** 2)
	cv2.imwrite(f"testFINAL.png", im)


def grayscale():
	im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	cv2.imwrite(f"grayscaled.png", im)


def sharpen(init):
	r = np.array([[0, -(init / 4), 0], [-(init / 4), init + 1, -(init / 4)], [0, -(init / 4), 0]])
	a = Kernel(r)
	# vis(a.matrix)

	im = convolve(img, a)
	cv2.imwrite(f"sharpened.png", im)


def motion_test():
	cv2.imwrite(f"motion_tested.png", convolve(img, Kernel(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [2, 1, 0, -1, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))))


def main():
	global parameter
	match mode[2:]:
		case "grayscale" | "gray":
			grayscale()
		case "inverse":
			inverse()
		case "edges":
			edges(float(parameter) if parameter is not None else 0.5)
		case "boxblur" | "blur":
			blur(float(parameter) if parameter is not None else 3)
		case "gaussianblur" | "gaussian":
			gaussian_blur(float(parameter) if parameter is not None else 1)
		case "sharpen":
			sharpen(float(parameter) if parameter is not None else 4)
		case "t":
			motion_test()
		case _:
			print("Invalid Mode")

main()

end = time()
cpu_end = process_time()

print(f"\n-----------------------\nWall-Clock: {(end - start):.5f}\nCPU: {(cpu_end - cpu_start):.5f}\n-----------------------\n")
