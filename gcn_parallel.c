#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#include "gcn_parallel.h"

GcnLayerParallel* create_gcn_layer_parallel(int input_dim, int output_dim) {
    GcnLayerParallel* layer = (GcnLayerParallel*)malloc(sizeof(GcnLayerParallel));
    if (!layer) return NULL;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->weights = (float*)aligned_alloc(32, (size_t)input_dim * output_dim * sizeof(float));
    if (!layer->weights) {
        free(layer);
        return NULL;
    }
    return layer;
}

void initialize_weights_random_parallel(GcnLayerParallel* layer) {
    for (int i = 0; i < layer->input_dim * layer->output_dim; ++i) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;  // [-1,1]
    }
}

void free_gcn_layer_parallel(GcnLayerParallel* layer) {
    if (layer) {
        free(layer->weights);
        free(layer);
    }
}

// Horizontal sum helper
static inline float hsum256(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, vlow);
    vlow = _mm_add_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

void gcn_forward_parallel(CsrGraph* graph, GcnLayerParallel* layer, float* input_features, float* output_features) {
    int num_nodes = graph->num_nodes;
    int in_dim = layer->input_dim;
    int out_dim = layer->output_dim;
    int* row_ptr = graph->row_ptr;
    int* col_idx = graph->col_idx;
    int* degrees = graph->degrees;

    float* transformed_features = (float*)malloc((size_t)num_nodes * out_dim * sizeof(float));
    if (!transformed_features) {
        return;
    }

    // Feature transformation (parallel over nodes)
    #pragma omp parallel for 
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            __m256 acc = _mm256_setzero_ps();
            int k;
            for (k = 0; k <= in_dim - 8; k += 8) {
                __m256 v_in = _mm256_loadu_ps(&input_features[i * in_dim + k]);
                __m256 v_w  = _mm256_loadu_ps(&layer->weights[k * out_dim + j]);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v_in, v_w));  // fallback instead of fmadd
            }
            float sum = hsum256(acc);
            for (; k < in_dim; ++k) {
                sum += input_features[i * in_dim + k] * layer->weights[k * out_dim + j];
            }
            transformed_features[i * out_dim + j] = sum;
        }
    }

    // Zero
    memset(output_features, 0, (size_t)num_nodes * out_dim * sizeof(float));

    // parallelized aggregation
    #pragma omp parallel for 
    for (int u = 0; u < num_nodes; ++u) {
        int start = row_ptr[u];
        int end = row_ptr[u + 1];
        for (int idx = start; idx < end; ++idx) {
            int v = col_idx[idx];
            float norm = 1.0f / sqrtf((float)degrees[u] * degrees[v]);
            for (int j = 0; j < out_dim; ++j) {
                output_features[u * out_dim + j] += norm * transformed_features[v * out_dim + j];
            }
        }
    }

    free(transformed_features);
}
