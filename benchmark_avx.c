#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <asm/unistd.h>
#include "graph.h"
#include "gcn_avx.h"
#define FEATURE_DIM 64  // parameterize later

#define CLOCK_SPEED 2.8
// Simple timing
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

// Load array from file
#include <stdio.h>
#include <stdlib.h>

// Loads an array of int from a .bin file
int* load_binary_array(const char* filename, int* length_out) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        return NULL;
    }

    // Seek to end to find file size
    if (fseek(f, 0, SEEK_END) != 0) {
        perror("fseek failed");
        fclose(f);
        return NULL;
    }

    long size_in_bytes = ftell(f);
    if (size_in_bytes == -1) {
        perror("ftell failed");
        fclose(f);
        return NULL;
    }

    rewind(f);  // Go back to start of file

    // Size in number of ints
    int num_elements = size_in_bytes / sizeof(int);

    int* array = (int*)malloc(size_in_bytes);
    if (!array) {
        perror("malloc failed");
        fclose(f);
        return NULL;
    }

    // Read entire file into array
    size_t read_elements = fread(array, sizeof(int), num_elements, f);
    if (read_elements != (size_t)num_elements) {
        perror("fread failed or incomplete read");
        free(array);
        fclose(f);
        return NULL;
    }

    fclose(f);

    *length_out = num_elements;
    return array;
}


int main() {
    srand((unsigned int)time(NULL));

    const int node_sizes[] = {1000, 10000, 100000, 1000000};
    const int num_node_sizes = sizeof(node_sizes) / sizeof(node_sizes[0]);

    FILE* csv = fopen("avx_results.csv", "w");
    if (!csv) {
        perror("Failed to open CSV file");
        return 1;
    }
    fprintf(csv, "NodeCount,GraphID,ExecutionTime(ms),Cycles,CPE\n");

    for (int ni = 0; ni < num_node_sizes; ++ni) {
        int num_nodes = node_sizes[ni];
        int num_graphs;
        if (num_nodes == 100000 || num_nodes == 1000000) {
            num_graphs = 5;
        } else {
            num_graphs = 10;
        }

        double total_time = 0.0;
        double total_cpe = 0.0;

        for (int gid = 0; gid < num_graphs; ++gid) {
            int newid = gid;
            if (num_nodes == 1000000) {
                newid = gid+5;
            }
            char row_ptr_filename[256], col_idx_filename[256];
            snprintf(row_ptr_filename, sizeof(row_ptr_filename),
                     "%d_nodes/erdos_renyi_%d_%d_row_ptr.bin", num_nodes, num_nodes, newid);
            snprintf(col_idx_filename, sizeof(col_idx_filename),
                     "%d_nodes/erdos_renyi_%d_%d_col_idx.bin", num_nodes, num_nodes, newid);

            printf("loading filename: %s\n", row_ptr_filename);
            int row_len, col_len;
            int* row_ptr = load_binary_array(row_ptr_filename, &row_len);
            int* col_idx = load_binary_array(col_idx_filename, &col_len);

            if (!row_ptr || !col_idx) {
                fprintf(stderr, "Failed to load binary CSR graph\n");
                return 1;
            }


            CsrGraph* graph = create_graph(num_nodes, col_len, row_ptr, col_idx);
            if (!graph) {
                free(row_ptr); free(col_idx);
                continue;
            }

            // Allocate features
            float* input_features = malloc((size_t)num_nodes * FEATURE_DIM * sizeof(float));
            float* output_features = malloc((size_t)num_nodes * FEATURE_DIM * sizeof(float));
            if (!input_features || !output_features) {
                perror("Failed to allocate features");
                free_graph(graph);
                free(row_ptr); free(col_idx);
                free(input_features); free(output_features);
                continue;
            }

            for (int i = 0; i < num_nodes * FEATURE_DIM; ++i) {
                input_features[i] = (float)rand() / RAND_MAX;
            }

            GcnLayerAvx* layer = create_gcn_layer_avx(FEATURE_DIM, FEATURE_DIM);
            initialize_weights_random_avx(layer);

            // Seup performance counter
            

            double start = get_time_ms();
            gcn_forward_avx(graph, layer, input_features, output_features);
            double end = get_time_ms();

            double exec_time_ms = end - start;
            double exec_time_sec = exec_time_ms / 1000.0;
            long long estimated_cycles = exec_time_sec * CLOCK_SPEED * 1e9;
            double cpe = (double)(estimated_cycles / (num_nodes * FEATURE_DIM));

            fprintf(csv, "%d,%d,%.4f,%lld,%.6f\n", num_nodes, gid, exec_time_sec, estimated_cycles, cpe);

            total_time += exec_time_sec;
            total_cpe += cpe;

            // Cleanup
            free_graph(graph);
            free(row_ptr);
            free(col_idx);
            free(input_features);
            free(output_features);
            free_gcn_layer_avx(layer);
        }

        double avg_time = total_time / num_graphs;
        double avg_cpe = total_cpe / num_graphs;
        printf("Average for %d nodes: ExecutionTime=%.4f ms, CPE=%.6f\n",
               num_nodes, avg_time, avg_cpe);
    }

    fclose(csv);
    return 0;
}
