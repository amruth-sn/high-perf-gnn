# high-perf-gnn

This repository contains code for high-performance graph neural network (GNN) implementations.

## Structure

- `include/`: Header files
  - Header files for graph structures, serial, AVX2, and OpenMP GCN algorithms
- `src/`: C source files
  - C implementation files for GNN algorithms and serial, AVX2, and OpenMP benchmarks
- `scripts/`: Python and shell scripts
  - Graph generation scripts
- `data/`: Data files
  - Output folder for benchmark results

  
## Generating Graphs

To generate smaller graphs run this command from root:

```bash
python3 scripts/generate_graphs.py
```
For larger graphs, run this command from root:

```bash
bash scripts/generate_large_graphs.sh
```

## Building Benchmarks

Run Makefile for these versions:

```bash
# Main Testing
make

# Serial
make serial

# AVX2-optimized
make avx

# Parallelized
make parallel
```