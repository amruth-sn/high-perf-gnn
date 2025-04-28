import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix
import os
from scipy.sparse import eye
def generate_erdos_renyi_graph(num_nodes=100000, param=0.001, seed=999, output_prefix="graph", save_binary=False):
    np.random.seed(seed)

    G = ig.Graph.Erdos_Renyi(n=num_nodes, p=param, directed=False, loops=False) # undirected w/o selfloops

    edges = np.array(G.get_edgelist())
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(row))

    # undirected
    row_full = np.concatenate([row, col])
    col_full = np.concatenate([col, row])
    data_full = np.ones(len(row_full))

    A = csr_matrix((data_full, (row_full, col_full)), shape=(num_nodes, num_nodes))

    # Add selfloops manually
    A += eye(num_nodes, format='csr')

    # CSR
    row_ptr = A.indptr
    col_idx = A.indices

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    if save_binary:
        row_ptr.astype(np.int32).tofile(f"{output_prefix}_row_ptr.bin")
        col_idx.astype(np.int32).tofile(f"{output_prefix}_col_idx.bin")
    else:
        np.savetxt(f"{output_prefix}_row_ptr.txt", row_ptr, fmt="%d")
        np.savetxt(f"{output_prefix}_col_idx.txt", col_idx, fmt="%d")

    print(f"Graph saved: {num_nodes} nodes, {A.nnz} edges (incl self-loops) [{output_prefix}]")

if __name__ == "__main__":
    for num_nodes in [1000, 10000]:
        os.makedirs(f"{num_nodes}_nodes", exist_ok=True)
        for i in range(10):  
            generate_erdos_renyi_graph(
                num_nodes=num_nodes,
                param=0.001,  # sparsity
                seed=i,
                output_prefix=f"{num_nodes}_nodes/erdos_renyi_{num_nodes}_{i}",
                save_binary=True,  
            )
