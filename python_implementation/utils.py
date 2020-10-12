import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

def visualize(g, t):
    G = nx.from_numpy_matrix(g)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, width=1.0, edge_cmap=plt.cm.Blues)
    plt.show()

def genEnv(filename):

    raw = np.loadtxt(filename, delimiter="\t", dtype=np.int)
    raw[:, :2] -= 1

    n = len(np.unique(raw[:, :2]))

    g = np.zeros([n,n], dtype=np.int8)

    for edge in raw:
        g[edge[0], edge[1]] = edge[2]

    return g

def updateT(t, evap):
    t = t * (1-evap)


def getInitalNodes(g):
    safe_init = np.sum(g, axis=1)
    return np.where(safe_init > 0)[0]

def displayHeatmap(t, i):
    fig, ax = plt.subplots()
    im = ax.imshow(t)


    # Loop over data dimensions and create text annotations.
    # for i in range(t.shape[0]):
    #     for j in range(t.shape[1]):
    #         text = ax.text(j, i, t[i, j], ha="center", va="center", color="w")

    ax.set_title("Pheromones")
    fig.tight_layout()
    plt.savefig("iter/%04d.png" % i)
    plt.close()