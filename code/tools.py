import cv2
from time import time

from kernels import *


def vis(matrix):  # for visualize
	if isinstance(matrix, Kernel):
		m = matrix.matrix
	elif isinstance(matrix, np.ndarray):
		m = matrix

	try:
		h, w, c = m.shape
	except ValueError:
		h, w, c = m.shape + tuple([1])

	if isinstance(matrix, Gaussian):
		max_val = m[(h - 1) // 2, (h - 1) // 2]
	else:
		max_val = np.max(m)

	im = cv2.merge([np.zeros(shape=(h, w, c), dtype=m.dtype), m, np.zeros(shape=(h, w, c), dtype=m.dtype)])

	for i in range(h):
		for j in range(w):
			if im[i, j][1] < 0:
				im[i, j] = [0, 0, abs(im[i, j][1])]

			im[i, j] = im[i, j] * (255 / max_val)

	cv2.imwrite(f"vis_{time()}.png", im)


def convolve(image, kernel: Kernel):
	try:
		h, w, c = image.shape
	except ValueError:
		h, w, c = image.shape + tuple([1])

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
