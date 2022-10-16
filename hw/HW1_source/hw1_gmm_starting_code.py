import numpy as np

"""
Generate data and initalize
We assume that there are two clusters
"""

x1 = np.random.normal(-10, 10, 2000)
x2 = np.random.normal(5, 5, 2000)
x: np.ndarray = np.concatenate((x1, x2), axis=0)
w_1 = 0.5
w_2 = 0.5
mu_1 = -2
mu_2 = 2
sigma_1 = 2
sigma_2 = 2

"""
Apply EM algorithm to update the GMM model
"""


def gaussian(sigma: float, mu: float) -> np.ndarray:
    global x
    return np.exp(-(x - mu) ** 2 / 2 * sigma ** 2) / np.sqrt(2 * np.pi) / sigma


for _ in range(10):
    # Step E
    # TODO next step here
    M = None
    pass
