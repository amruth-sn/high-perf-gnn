#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

#include "graph.h"
#include "gcn.h"

void print_features(const char* title, float* features, int num_nodes, int feature_dim) {
    printf("%s (%d nodes, %d features):\n", title, num_nodes, feature_dim);
    for (int i = 0; i < num_nodes; ++i) {
        printf("  Node %d: [", i);
        for (int j = 0; j < feature_dim; ++j) {
            printf("%.4f%s", features[i * feature_dim + j], (j == feature_dim - 1) ? "" : ", ");
        }
        printf("]\n");
    }
    printf("\n");
}

int main() {
    // for weight initialization
    srand((unsigned int)time(NULL));

    // nodes: 4
    // edges: 0-1, 1-0, 1-2, 2-1, 2-0, 0-2, 1-3, 3-1 
    // undirected graphs have doubled edges bc csr is symmetric
    int num_nodes = 4;
    int num_edges = 8; 

    // CSR representation
    int row_ptr[] = {0, 2, 5, 7, 8}; 
    
    // node 0 neighbors: 0 to 1
    // node 1 neighbors: 2 to 4
    // node 2 neighbors: 5 to 6
    // node 3 neighbors: 7 to 7
    int col_idx[] = {1, 2,  // neighbors of 0
                     0, 2, 3,  // neighbors of 1
                     0, 1,     // neighbors of 2
                     1};       // neighbors of 3

    // Create the graph structure
    CsrGraph* graph = create_graph(num_nodes, num_edges, row_ptr, col_idx);
    if (!graph) {
        return 1;
    }

    printf("Graph created: %d nodes, %d edges (CSR entries)\n", graph->num_nodes, graph->num_edges);
    printf("Node degrees (incl. self-loop):\n");
    for(int i=0; i<graph->num_nodes; ++i) {
        printf("  Node %d: %d\n", i, graph->degrees[i]);
    }
    printf("\n");


    int input_feature_dim = 2; 
    float* input_features = (float*)malloc((size_t)num_nodes * input_feature_dim * sizeof(float));
    if (!input_features) {
        free_graph(graph);
        return 1;
    }
    //example
    float initial_data[][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}};
    for(int i=0; i<num_nodes; ++i) {
        for(int j=0; j<input_feature_dim; ++j) {
            input_features[i * input_feature_dim + j] = initial_data[i][j];
        }
    }

    print_features("Initial Features", input_features, num_nodes, input_feature_dim);

    int output_feature_dim = 3;
    GcnLayer* gcn_layer = create_gcn_layer(input_feature_dim, output_feature_dim);
    if (!gcn_layer) {
        free(input_features);
        free_graph(graph);
        return 1;
    }
    initialize_weights_random(gcn_layer);
    printf("GCN Layer created: Input Dim=%d, Output Dim=%d\n\n", gcn_layer->input_dim, gcn_layer->output_dim);

    float* output_features = (float*)malloc((size_t)num_nodes * output_feature_dim * sizeof(float));
    if (!output_features) {
        free_gcn_layer(gcn_layer);
        free(input_features);
        free_graph(graph);
        return 1;
    }

    gcn_forward(graph, gcn_layer, input_features, output_features);

    print_features("Output Features (After 1 GCN Layer)", output_features, num_nodes, output_feature_dim);

    // 7. Cleanup
    free_graph(graph);
    free(input_features);
    free_gcn_layer(gcn_layer);
    free(output_features);

    return 0;
}
