#include "graph.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Calculates degree + 1 for self-loop normalization
void calculate_degrees(CsrGraph* graph) {
    if (!graph) return;
    graph->degrees = (int*)malloc(graph->num_nodes * sizeof(int));
    if (!graph->degrees) {
        perror("Failed to allocate memory for degrees");
        return; 
    }
    // Initialize degrees to 1 (for self-loop)
    for (int i = 0; i < graph->num_nodes; ++i) {
        graph->degrees[i] = 1;
    }
    // Add degrees from actual edges (assuming undirected, edges counted once)
    // CSR stores all neighbors, so the difference gives the degree
    for (int i = 0; i < graph->num_nodes; ++i) {
        graph->degrees[i] += graph->row_ptr[i+1] - graph->row_ptr[i];
    }
}

CsrGraph* create_graph(int num_nodes, int num_edges, int *row_ptr_in, int *col_idx_in) {
    if (num_nodes <= 0 || num_edges < 0 || !row_ptr_in || (num_edges > 0 && !col_idx_in)) {
        fprintf(stderr, "Invalid input for create_graph\n");
        return NULL;
    }

    CsrGraph *graph = (CsrGraph*)malloc(sizeof(CsrGraph));
    if (!graph) {
        perror("Failed to allocate memory for graph structure");
        return NULL;
    }

    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    graph->degrees = NULL; 

    // Allocate and copy row_ptr
    graph->row_ptr = (int*)malloc((num_nodes + 1) * sizeof(int));
    if (!graph->row_ptr) {
        perror("Failed to allocate memory for row_ptr");
        free(graph);
        return NULL;
    }
    memcpy(graph->row_ptr, row_ptr_in, (num_nodes + 1) * sizeof(int));

    // Allocate and copy col_idx
    if (num_edges > 0) {
        graph->col_idx = (int*)malloc(num_edges * sizeof(int));
        if (!graph->col_idx) {
            perror("Failed to allocate memory for col_idx");
            free(graph->row_ptr);
            free(graph);
            return NULL;
        }
        memcpy(graph->col_idx, col_idx_in, num_edges * sizeof(int));
    } else {
        graph->col_idx = NULL; // No edges
    }

    // Precompute degrees (including self-loops)
    calculate_degrees(graph);
    if (!graph->degrees) { // Check if degree calculation failed
        free_graph(graph);
        return NULL;
    }

    return graph;
}

void free_graph(CsrGraph* graph) {
    if (graph) {
        free(graph->row_ptr);
        free(graph->col_idx);
        free(graph->degrees);
        free(graph);
    }
} 