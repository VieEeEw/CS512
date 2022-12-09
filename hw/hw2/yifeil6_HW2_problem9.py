import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import networkx as nx

from functools import lru_cache
import random

DATA_NAME = "social_network.txt"


class SISModel:
    def __init__(self, adj: np.ndarray, beta: float = 0.01, delta: float = 0.05):
        self.iteration: int = 0
        self.adj: np.ndarray = adj
        self.num_nodes: int = adj.shape[0]
        self.state: np.ndarray = np.ones(self.num_nodes)
        self.beta: float = beta
        self.delta: float = delta
        self.sys_mat = (1 - self.delta) * np.identity(self.num_nodes) + self.beta * self.adj

    @lru_cache(None)
    def prob(self, n: int) -> float:
        return 1 - (1 - self.beta) ** n

    def _step(self) -> int:
        self.iteration += 1
        self.state = np.array(
            [self.state[i] or self.prob(self.state @ self.adj[i, :]) >= random.random() for i in range(self.num_nodes)])
        for i in range(self.num_nodes):
            if self.state[i]:
                self.state[i] = random.random() > self.delta
        return sum(self.state)

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

    random.seed(114514)
    adj_matrix = nx.to_numpy_array(G.subgraph(largest_cc))
    sis = SISModel(adj_matrix)
    sis.present(300, "c", 100, save=True)
    sis = SISModel(adj_matrix, delta=0.4)
    sis.present(30, "d", 0, save=True)
