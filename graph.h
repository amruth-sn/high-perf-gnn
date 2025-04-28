#ifndef GRAPH_H
#define GRAPH_H

#include <stddef.h> // For size_t

// CSR graph
typedef struct {
    int num_nodes;
    int num_edges;
    int *row_ptr;  // Size: num_nodes + 1
    int *col_idx;  // Size: num_edges
    int *degrees;  // Size: num_nodes 
} CsrGraph;

CsrGraph* create_graph(int num_nodes, int num_edges, int *row_ptr, int *col_idx);
void free_graph(CsrGraph* graph);
void calculate_degrees(CsrGraph* graph); 

#endif // GRAPH_H 