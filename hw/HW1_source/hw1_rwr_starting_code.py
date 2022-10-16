import numpy as np
import networkx as nx
import random

G = nx.read_edgelist("email-Eu-core.txt")
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


def random_walk(query: int, adj_mat_norm: np.ndarray):
    """Implement random walk
    """
    r = np.zeros(adj_mat_norm.shape[0])
    r[query] = 1
    for _ in range(max_iter):
        r = adj_mat_norm @ r
    return r


random.seed(0)


def random_walk_restart(query: int, adj_mat_norm: np.ndarray, c: float = 0.9):
    """Implement random walk with restart
    """
    r = np.zeros(adj_mat_norm.shape[0])
    r[query] = 1
    e = r.copy()
    for _ in range(max_iter):
        r = c * adj_mat_norm @ r + (1 - c) * e
    return r


for q in random.choices(range(700), k=2):
    print("Random walk:")
    print(random_walk(q, A_norm))
    print("Random walk with restart")
    print(random_walk_restart(q, A_norm))
