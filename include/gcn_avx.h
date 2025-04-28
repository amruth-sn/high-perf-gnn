#ifndef GCN_AVX_H
#define GCN_AVX_H

#include "graph.h"

typedef struct {
    int input_dim;
    int output_dim;
    float* weights;  
} GcnLayerAvx;

GcnLayerAvx* create_gcn_layer_avx(int input_dim, int output_dim);
void initialize_weights_random_avx(GcnLayerAvx* layer);
void free_gcn_layer_avx(GcnLayerAvx* layer);

void gcn_forward_avx(CsrGraph* graph, GcnLayerAvx* layer, float* input_features, float* output_features);

#endif // GCN_AVX_H
