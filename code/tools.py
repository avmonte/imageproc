import cv2
from time import time

import numpy as np

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


def fastconv(image: np.ndarray, kernel: Kernel):

	i_f = np.flip(image.flatten())
	k_f = np.flip(kernel.matrix.flatten())

	new_degree = len(i_f) + len(k_f) - 2

	x = np.array([i for i in range(new_degree + 1)])
	i_y = np.array([i_f[0]])
	k_y = np.array([k_f[0]])

	for i in x:
		i_y = np.append(i_y, np.polyval(i_f, i))
		k_y = np.append(k_y, np.polyval(k_f, i))

	y = i_y * k_y
	p_f = np.array(np.polyfit(x, y, new_degree))[:len(i_f)]

	return p_f.reshape(i_f.shape)


