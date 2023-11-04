import cv2
from time import time

import numpy as np


def vis(matrix):  # for visualize
	im = cv2.merge([np.zeros(shape=matrix.shape, dtype=matrix.dtype), matrix, np.zeros(shape=matrix.shape, dtype=matrix.dtype)])
	h, w, c = im.shape

	for i in range(h):
		for j in range(w):
			if im[i, j][1] < 0:
				im[i, j] = [0, 0, abs(im[i, j][1])]

			im[i, j] = im[i, j] * (255 / np.max(matrix))
			print(im[i, j])

	cv2.imwrite(f"vis_{time()}.png", im)
