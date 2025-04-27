import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

def generate_erdos_renyi_graph(num_nodes=1000, param=0.01, seed=999, output_prefix="graph"):
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(num_nodes, param, seed=seed)
    #rm generated self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    #undirected
    G = G.to_undirected()

    #csr
    A = nx.to_scipy_sparse_array(G, format='csr')

    #add explicit self loops
    A = A + csr_matrix(np.eye(A.shape[0]))

    row_ptr = A.indptr
    col_idx = A.indices
    np.savetxt(f"{output_prefix}_row_ptr.txt", row_ptr, fmt="%d")
    np.savetxt(f"{output_prefix}_col_idx.txt", col_idx, fmt="%d")
    print(f"Graph saved with {num_nodes} nodes and {A.nnz} total edges (incl. self-loops)")


if __name__ == "__main__":
    generate_erdos_renyi_graph(
        num_nodes=1000,
        param=0.01,
    )