#include "gcn.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> 

// relu
float relu(float x) {
    return x > 0 ? x : 0;
}

GcnLayer* create_gcn_layer(int input_dim, int output_dim) {
    if (input_dim <= 0 || output_dim <= 0) {
        return NULL;
    }
    GcnLayer *layer = (GcnLayer*)malloc(sizeof(GcnLayer));
    if (!layer) {
        return NULL;
    }
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->weights = (float*)malloc((size_t)input_dim * output_dim * sizeof(float));
    if (!layer->weights) {
        free(layer);
        return NULL;
    }
    return layer;
}

void free_gcn_layer(GcnLayer* layer) {
    if (layer) {
        free(layer->weights);
        free(layer);
    }
}

// Simple random weight initialization (Glorot/Xavier style often better)
void initialize_weights_random(GcnLayer* layer) {
    if (!layer || !layer->weights) return;
    size_t num_weights = (size_t)layer->input_dim * layer->output_dim;
    // Basic random initialization between -0.5 and 0.5
    for (size_t i = 0; i < num_weights; ++i) {
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
}

// GCN forward pass
void gcn_forward(
    CsrGraph* graph,
    GcnLayer* layer,
    float* input_features,
    float* output_features 
) {
    if (!graph || !layer || !input_features || !output_features) {
        return;
    }

    int num_nodes = graph->num_nodes;
    int in_dim = layer->input_dim;
    int out_dim = layer->output_dim;

    // H' = HW
    float* transformed_features = (float*)calloc((size_t)num_nodes * out_dim, sizeof(float));
    if (!transformed_features) {
        return;
    }

    for (int i = 0; i < num_nodes; ++i) { //nodewise
        for (int j = 0; j < out_dim; ++j) { 
            float sum = 0.0f;
            for (int k = 0; k < in_dim; ++k) { 
                sum += input_features[i * in_dim + k] * layer->weights[k * out_dim + j];
            }
            transformed_features[i * out_dim + j] = sum;
        }
    }

    // Aggregation
    // init output features to zero
    memset(output_features, 0, (size_t)num_nodes * out_dim * sizeof(float));

    for (int u = 0; u < num_nodes; ++u) {
        int start = graph->row_ptr[u];
        int end = graph->row_ptr[u + 1];
        for (int edge_idx = start; edge_idx < end; ++edge_idx) {
            int v = graph->col_idx[edge_idx]; // Neighbor node

            float norm_uv = 1.0f / sqrtf((float)graph->degrees[u] * graph->degrees[v]);

            for (int k = 0; k < out_dim; ++k) {
                output_features[u * out_dim + k] += norm_uv * transformed_features[v * out_dim + k];
            }
        }
    }

    // Activation (ReLU)
    for (int i = 0; i < num_nodes * out_dim; ++i) {
        output_features[i] = relu(output_features[i]);
    }

    free(transformed_features);
} 
