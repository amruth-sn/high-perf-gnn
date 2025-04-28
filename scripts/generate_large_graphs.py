import sys
import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix, eye
import os

def generate_erdos_renyi_graph(num_nodes=100000, param=0.001, seed=999, output_prefix="graph", save_binary=True):
    np.random.seed(seed)

    G = ig.Graph.Erdos_Renyi(n=num_nodes, p=param, directed=False, loops=False)

    edges = np.array(G.get_edgelist())
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(row))

    row_full = np.concatenate([row, col])
    col_full = np.concatenate([col, row])
    data_full = np.ones(len(row_full))

    A = csr_matrix((data_full, (row_full, col_full)), shape=(num_nodes, num_nodes))

    A += eye(num_nodes, format='csr')

    row_ptr = A.indptr
    col_idx = A.indices

    os.makedirs(f"{num_nodes}_nodes", exist_ok=True)

    if save_binary:
        row_ptr.astype(np.int32).tofile(f"{num_nodes}_nodes/erdos_renyi_{num_nodes}_{seed}_row_ptr.bin")
        col_idx.astype(np.int32).tofile(f"{num_nodes}_nodes/erdos_renyi_{num_nodes}_{seed}_col_idx.bin")
    else:
        np.savetxt(f"{num_nodes}_nodes/erdos_renyi_{num_nodes}_{seed}_row_ptr.txt", row_ptr, fmt="%d")
        np.savetxt(f"{num_nodes}_nodes/erdos_renyi_{num_nodes}_{seed}_col_idx.txt", col_idx, fmt="%d")

    print(f"Graph saved: {num_nodes} nodes, seed {seed}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_graph_task.py <num_nodes> <seed>")
        sys.exit(1)

    num_nodes = int(sys.argv[1])
    seed = int(sys.argv[2])

    generate_erdos_renyi_graph(num_nodes=num_nodes, seed=seed,
                                output_prefix=f"{num_nodes}_nodes/erdos_renyi_{num_nodes}_{seed}")
