# High-Performance Graph Neural Networks

Benchmarking high-performance graph convolutional network (GCN) implementations.

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

To generate smaller graphs, run this command from root:

```bash
python3 scripts/generate_graphs.py
```
For larger graphs, run this command from root on a Sun Grid Engine:

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

# OpenMP-Parallelized
make parallel
```

## Considerations

Generated graph row/col arrays for node sizes >10^5^ are **large**. Make sure you have enough storage space!

Feel free to play around with parameters for graph sparsity, edge probabilities, node counts, etc. I'm doing this too!

## Current Work

I'm working on benchmarking the GPU implementation of this simple GCN using NVIDIA's [CUDA](https://developer.nvidia.com/cuda-toolkit) library.

I also want to extend these algorithms to N-layer networks, parameterized feature dimensions, and ultimately work towards a backpropagation implementation for sparse graph operations to learn the weight matrix!

## Learn More

Read `report.pdf`! :)
