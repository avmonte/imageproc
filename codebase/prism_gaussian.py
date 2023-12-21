
from tools import vis

import numpy as np


class KernelGenerator:
    def __init__(self, stdev):
        self.stdev = stdev

    def generate(self, alpha):
        threshold = 0.97
        alpha_rad = np.radians(alpha)  # Convert alpha to radians
        a = np.cos(alpha_rad)
        b = np.sin(alpha_rad)

        amp = 1 / 100

        gaussian = lambda x, y: amp * np.exp(-((a * x + b * y) ** 2) / (2 * self.stdev ** 2))

        n = 7
        arr = np.zeros(shape=(n, n, 1), dtype=np.float64)

        while np.sum(arr, dtype=np.float64) < threshold:
            n += 2
            arr = np.zeros(shape=(n, n, 1), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    x, y = i - (n // 2), j - (n // 2)
                    arr[i, j] = gaussian(x, y)

        print(np.sum(arr, dtype=np.float64), n)  # for debug
        return arr / np.sum(arr, dtype=np.float64)  # normalize


kernel = KernelGenerator(3).generate(0)

vis(kernel)
