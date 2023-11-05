import cv2
from time import time

import numpy as np


def vis(matrix):  # for visualize
	h, w, c = matrix.shape
	center = (h - 1) // 2
	max_val = matrix[center, center]
	im = cv2.merge([np.zeros(shape=(h, w, c), dtype=matrix.dtype), matrix, np.zeros(shape=(h, w, c), dtype=matrix.dtype)])

	for i in range(h):
		for j in range(w):
			if im[i, j][1] < 0:
				im[i, j] = [0, 0, abs(im[i, j][1])]

			im[i, j] = im[i, j] * (255 / max_val)
			np.max()

	cv2.imwrite(f"vis_{time()}.png", im)
