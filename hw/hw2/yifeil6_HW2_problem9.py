import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import networkx as nx

DATA_NAME = "social_network.txt"
THRESHOLD = 0.4


class SISModel:
    def __init__(self, adj: np.ndarray, threshold: float = THRESHOLD, beta: float = 0.01, delta: float = 0.05):
        self.iteration: int = 0
        self.threshold: float = threshold
        self.adj: np.ndarray = adj
        self.num_nodes: int = adj.shape[0]
        self.state: np.ndarray = np.ones(self.num_nodes)
        self.beta: float = beta
        self.delta: float = delta
        self.sys_mat = (1 - self.delta) * np.identity(self.num_nodes) + self.beta * self.adj

    def _step(self) -> int:
        self.iteration += 1
        self.state = self.sys_mat @ self.state
        self.state[self.state > 1] = 1
        return sum(self.state >= self.threshold)

    def _run(self, steps: int) -> list:
        ret = []
        for _ in range(steps):
            infected = self._step()
            ret.append(infected)
        return ret

    def present(self, steps: int, q: str, show_after=100, save=False):
        infected = self._run(steps)[show_after:]
        plt.plot(range(show_after, steps), infected, "-", label="Infected")
        plt.plot(range(show_after, steps), [self.num_nodes - i for i in infected], "-", label="Healthy")
        plt.title(f"hw2-9-{q}")
        plt.xlabel("Time Step")
        plt.ylabel("# of Infected")
        plt.legend()
        if save:
            plt.savefig(f"latex/figs/hw2-9-{q}.png")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    file = glob(DATA_NAME) + glob(os.path.join("*", DATA_NAME))
    if not file:
        print(f"Data file {DATA_NAME} is not found anywhere, please check.")
        exit(0)
    G = nx.read_edgelist(file[0], delimiter=",")
    largest_cc = max(nx.connected_components(G), key=len)

    adj_matrix = nx.to_numpy_array(G.subgraph(largest_cc))
    sis = SISModel(adj_matrix)
    sis.present(500, "c", 100)
    sis = SISModel(adj_matrix, delta=0.4)
    sis.present(10, "d", 0)
