#ifndef GCN_PARALLEL_H
#define GCN_PARALLEL_H

#include "graph.h"

typedef struct {
    int input_dim;
    int output_dim;
    float* weights;  // (input_dim x output_dim)
} GcnLayerParallel;

GcnLayerParallel* create_gcn_layer_parallel(int input_dim, int output_dim);
void initialize_weights_random_parallel(GcnLayerParallel* layer);
void free_gcn_layer_parallel(GcnLayerParallel* layer);

void gcn_forward_parallel(CsrGraph* graph, GcnLayerParallel* layer, float* input_features, float* output_features);

#endif // GCN_PARALLEL_H
