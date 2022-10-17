import numpy as np
import networkx as nx
import sys
import random
from glob import glob

np.set_printoptions(suppress=True)
FILE_NAME = "email-Eu-core.txt"

file = glob(f"{FILE_NAME}") + glob(f"*/{FILE_NAME}")

if len(file) == 0:
    print(f"Cannot find data file {FILE_NAME} in any subdirectory, please check")
    exit(0)
G = nx.read_edgelist(file[0])
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)
A: np.ndarray = nx.to_numpy_array(G)
max_iter = 300
"""
Normalize the given adjacency matrix
"""
D = np.zeros(A.shape)
mask = np.ones(A.shape[0])
for i in range(A.shape[0]):
    deg = A[i, :] @ mask
    D[i, i] = (1 / deg) ** 0.5 if deg != 0 else 0
A_norm = D @ A @ D
random.seed(0)


def random_walk(query: int, adj_mat_norm: np.ndarray):
    """Implement random walk
    """
    r = np.zeros(adj_mat_norm.shape[0])
    r[query] = 1
    for _ in range(max_iter):
        r = adj_mat_norm @ r
    return np.round(r.astype(np.float64), 6)


def random_walk_restart(query: int, adj_mat_norm: np.ndarray, c: float = 0.9):
    """Implement random walk with restart
    """
    r = np.zeros(adj_mat_norm.shape[0])
    r[query] = 1
    e = r.copy()
    for _ in range(max_iter):
        r = c * adj_mat_norm @ r + (1 - c) * e
    return np.round(r.astype(np.float64), 6)


if __name__ == "__main__":
    with open("HW1.8Results.txt", "w") as f:
        print("Result written to HW1.8Results.txt")
        sys.stdout = f
        for q in random.choices(range(700), k=2):
            print(f"Random walk without restart, initial query index {q}")
            print(random_walk(q, A_norm))
            print(f"Random walk with restart, initial query index {q}")
            print(random_walk_restart(q, A_norm))
