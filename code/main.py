from sys import argv
from time import process_time, time
import argparse

# Time
start = time()
cpu_start = process_time()

# Read
path = argv[1]
mode = argv[2]
try:
	parameter = argv[3]
except IndexError:
	parameter = None

try:
	name = argv[4]
except IndexError:
	name = None

# Load
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
	return img


def test():
	return convolve(img, Kernel(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [2, 1, 0, -1, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))


def main():
	global parameter
	param = lambda x: float(parameter) if parameter is not None else x
	match mode[2:]:
		case "grayscale" | "gray":
			p = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		case "inverse":
			p = inverse()
		case "edges":
			p = np.sqrt(convolve(img, Edge(param(0.5))) ** 2 + convolve(img, Edge(param(0.5), False)) ** 2)
		case "boxblur" | "blur":
			p = convolve(img, Box(param(3)))
			# p = fastconv(img, Box(param(3)))
		case "gaussianblur" | "gaussian":
			p = convolve(img, Gaussian(float(parameter) if parameter is not None else 1))
		case "sharpen":
			p = convolve(img, Sharpen(param(4)))
		case "t":
			p = test()
		case _:
			print("Invalid Mode")
			return

	save_filename =  f"{mode[2:]}_{path.split('/')[-1]}"
	if name is not None:
		save_filename = name

	cv2.imwrite(save_filename, p)



main()

# Time Stats
print(f"\n{'-' * 23}\nWall-Clock: {(time() - start):.5f}\nCPU: {(process_time() - cpu_start):.5f}\n{'-' * 23}\n")
