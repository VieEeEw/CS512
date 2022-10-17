import numpy as np
from numpy import linalg
import networkx as nx
from glob import glob
FILE_NAME = "email-Eu-core.txt"

if __name__ == "__main__":
    file = glob(f"{FILE_NAME}") + glob(f"*/{FILE_NAME}")

    if len(file) == 0:
        print(f"Cannot find data file {FILE_NAME} in any subdirectory, please check")
        exit(0)
    G = nx.read_edgelist(file[0])
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc)
    A = nx.to_numpy_array(G)

    """
    Eigen decompose the given adjacency matrix
    Conduct spectral partition
    """
    L = np.diag(np.sum(A, axis=1)) - A
    eig = linalg.eigh(L)
    q = np.round(sorted(zip(eig[0], eig[1].T), key=lambda x: x[0])[1][1].astype(np.float64), 5)

    PB = q >= 0
    PA = q < 0
    print(f"Cluster-A size: {np.sum(PA)}.\tCluster-B size: {np.sum(PB)}.\t# of cuts: {np.sum(A[PA, :][:, PB]) * 2}")
