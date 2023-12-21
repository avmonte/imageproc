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


# generic Fast Fourier Transform
def FFT(x: np.ndarray):
	n = len(x)
	if n == 1:
		return x

	even = FFT(x[::2])
	odd = FFT(x[1::2])
	omega = lambda n: np.exp(2*np.pi*1j/n)

	points = [0] * n
	for e in range(n // 2):
		points[e] = even[e] + omega(n) ** e * odd[e]
		points[e + n // 2] = even[e] - omega(n) ** e * odd[e]

	return np.array(points)


# generic Inverse Fast Fourier Transform
def IFFT(x: np.ndarray):
	n = len(x)
	if n == 1:
		return x

	even = IFFT(x[::2])
	odd = IFFT(x[1::2])
	omega = lambda n: np.exp(2*np.pi*1j/n)

	points = [0] * n
	for e in range(n // 2):
		points[e] = even[e] + omega(n) ** -e * odd[e]
		points[e + n // 2] = even[e] - omega(n) ** -e * odd[e]

	return np.array(points)


def ineff_mult(x: np.ndarray, y: np.ndarray):
	# inefficient multiplication
	n = len(x)
	m = len(y)
	z = np.zeros(n + m - 1, dtype=np.complex128)

	for i in range(n):
		for j in range(m):
			z[i + j] += x[i] * y[j]

	return z


coef_p1 = np.array([1] * 70)  # 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 6x^5
coef_p2 = np.array([2] * 70)

start = time()
print(ineff_mult(coef_p1, coef_p2))
print(f"RUNTIME: {time() - start}")

start = time()
val_p1 = FFT(coef_p1)
val_p2 = FFT(coef_p2)
print(IFFT(val_p1 * val_p2))
print(f"RUNTIME: {time() - start}")
# time complexity: O(n^2) -> O(nlogn) ?


def fast_convolve(image: np.ndarray, kernel: Kernel):
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
