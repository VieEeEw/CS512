import numpy as np
import sys

"""
Generate data and initalize
We assume that there are two clusters
"""

x1 = np.random.normal(-10, 10, 2000)
x2 = np.random.normal(5, 5, 2000)
n = 4000
x = np.concatenate((x1, x2), axis=0)
w_1 = 0.5
w_2 = 0.5
mu_1 = -2
mu_2 = 2
sigma_1 = 4
sigma_2 = 4

"""
Apply EM algorithm to update the GMM model
"""


def gaussian(sigma: float, mu: float) -> np.ndarray:
    global x
    return np.exp(-(x - mu) ** 2 / (2 * sigma)) / np.sqrt(2 * np.pi * sigma)


if __name__ == "__main__":
    with open("HW1.9Results.txt", "w") as f:
        print("Result written to HW1.9Results.txt")
        sys.stdout = f
        for i in range(10):
            # Step E
            m1 = w_1 * gaussian(sigma_1, mu_1)
            m2 = w_2 * gaussian(sigma_2, mu_2)
            denominator = m1 + m2
            m1 /= denominator
            m2 /= denominator
            # Step M
            s1 = np.sum(m1)
            s2 = np.sum(m2)
            mu_1 = m1 @ x / s1
            mu_2 = m2 @ x / s2
            sigma_1 = m1 @ (x - mu_1) ** 2 / s1
            sigma_2 = m2 @ (x - mu_2) ** 2 / s2
            w_1 = m1 / n
            w_2 = m2 / n
            print(
                f"--------------------iterator {i + 1}-------------------- \nmu1: {mu_1}, mu2: {mu_2}, sigma1: {np.sqrt(sigma_1)}, sigma1: {np.sqrt(sigma_2)}\nw1:\n{w_1}\nw2:\n{w_2}")
