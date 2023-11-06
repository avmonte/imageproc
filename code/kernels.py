import numpy as np


class Kernel:
	# check the two props

	def __init__(self, matrix):
		self.matrix = matrix
		self.size = matrix.shape[0] // 2
		self.coef = 1 / np.sum(matrix) if np.sum(matrix) != 0 else 1

	def __str__(self):
		this = ''
		for i in self.matrix:
			for j in i:
				this += f"[{j[0]:.5f}] "
			this += '\n'

		return this


class EdgeKernel(Kernel):
	def __init__(self, init, horizontal=True):
		self.init = init
		self.horizontal = horizontal
		super().__init__(self.generate())

	def generate(self):
		arr = np.array([[-(self.init / 2), -self.init, -(self.init / 2)], [0, 0, 0], [self.init / 2, self.init, self.init / 2]])
		if self.horizontal:
			return arr
		else:
			return arr.transpose()


class Gaussian(Kernel):

	def __init__(self, stdev):
		self.stdev = stdev
		super().__init__(self.generate())

	def generate(self):
		threshold = 0.95
		gaussian = lambda x, y: (np.exp(-(x ** 2 + y ** 2) / (2 * (self.stdev ** 2))) / (2 * np.pi * (self.stdev ** 2)))
		n = 3
		arr = np.zeros(shape=(n, n, 1), dtype=np.float64)

		while np.sum(arr, dtype=np.float64) < threshold:
			n += 2
			arr = np.zeros(shape=(n, n, 1), dtype=np.float16)
			for i in range(n):
				for j in range(n):
					arr[i, j] = gaussian(i - (n // 2), j - (n // 2))

		print(np.sum(arr, dtype=np.float64), n)  # for debug
		return arr
