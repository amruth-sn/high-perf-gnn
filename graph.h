#ifndef GRAPH_H
#define GRAPH_H

#include <stddef.h> // For size_t

// Compressed Sparse Row (CSR) graph representation
typedef struct {
    int num_nodes;
    int num_edges;
    int *row_ptr;  // Size: num_nodes + 1
    int *col_idx;  // Size: num_edges
    int *degrees;  // Size: num_nodes (precomputed degrees + 1 for self-loops)
} CsrGraph;

CsrGraph* create_graph(int num_nodes, int num_edges, int *row_ptr, int *col_idx);
void free_graph(CsrGraph* graph);
void calculate_degrees(CsrGraph* graph); // Calculates degree + 1 for each node

#endif // GRAPH_H 