import numpy as np
import scipy.signal as ss


def conv2d_stride(arr: np.ndarray, kernel: np.ndarray, stride: int = 1, mode: str = "valid") -> np.ndarray:
    return ss.convolve2d(arr, kernel[::-1, ::-1], mode=mode)[::stride, ::stride]


if __name__ == "__main__":
    mat = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1]).reshape(5, 5)
    k1 = np.array([0, 1, 1, 0]).reshape(2, 2)
    k2 = np.array([1, 0, 1, 1]).reshape(2, 2)
    avg_pool = np.array([0.25, 0.25, 0.25, 0.25]).reshape(2, 2)  # avg pooling
    a1 = ss.convolve2d(mat, k1, mode="valid")
    a2 = ss.convolve2d(mat, k2, mode="valid")
    # 2-5 a
    print(a1.tolist())
    print(a2.tolist())
    # 2-5 b
    print(conv2d_stride(a1, avg_pool, 2).tolist())
    print(conv2d_stride(a2, avg_pool, 2).tolist())
    # 2-5 d
    h0 = np.array([1, 1]).reshape(2, 1)
    x1 = np.array([1, 0]).reshape(2, 1)
    x2 = np.array([1, 1]).reshape(2, 1)
    W = np.array([1, 1, 0, 1]).reshape(2, 2)
    U = np.array([0, 1, 1, 0]).reshape(2, 2)
    V = np.array([1, 1, 1, 0]).reshape(2, 2)
    h1 = W @ h0 + U @ x1
    h2 = W @ h1 + U @ x2
    y2 = V @ h2
    print(h1)
    print(h2)
    print(y2)
