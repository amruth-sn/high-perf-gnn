#ifndef GCN_H
#define GCN_H

#include "graph.h"

// GCN Layer
typedef struct {
    int input_dim;
    int output_dim;
    float *weights; // flattened weight matrix (input_dim * output_dim)
    // TODO: add bias
} GcnLayer;

GcnLayer* create_gcn_layer(int input_dim, int output_dim);
void free_gcn_layer(GcnLayer* layer);
void initialize_weights_random(GcnLayer* layer); 

// forward pass for one layer
void gcn_forward(
    CsrGraph* graph,
    GcnLayer* layer,
    float* input_features,  // flattened input features (num_nodes * input_dim)
    float* output_features // flattened output features (num_nodes * output_dim)
);

#endif // GCN_H 