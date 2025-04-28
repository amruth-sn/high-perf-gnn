CC = nvcc
CFLAGS = -arch compute_70 -code sm_70

SRC = src/main_testing.c src/gcn.c src/graph.c
OBJ = $(SRC:.c=.o)
EXEC = gcn_program

CUDA_FLAGS = -O3  

INCLUDES = -I./include/

LDFLAGS =

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $(EXEC) $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(OBJ) $(EXEC) benchmark_serial benchmark_avx benchmark_parallel benchmark_parallel_*

serial:
	gcc -O2 -o benchmark_serial src/benchmark_serial.c src/graph.c src/gcn.c -lm -I./include/

serial_clean:
	rm -f benchmark_serial

avx:
	gcc -O2 -mavx2 -o benchmark_avx src/benchmark_avx.c src/gcn_avx.c src/graph.c -lm -I./include/

avx_clean:
	rm -f benchmark_avx

parallel:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel src/benchmark_parallel.c src/gcn_parallel.c src/graph.c -lm -I./include/
	export OMP_NUM_THREADS=1

parallel_2:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_2 src/benchmark_parallel.c src/gcn_parallel.c src/graph.c -lm -I./include/
	export OMP_NUM_THREADS=2

parallel_4:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_4 src/benchmark_parallel.c src/gcn_parallel.c src/graph.c -lm -I./include/
	export OMP_NUM_THREADS=4

parallel_8:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_8 src/benchmark_parallel.c src/gcn_parallel.c src/graph.c -lm -I./include/
	export OMP_NUM_THREADS=8

parallel_16:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_16 src/benchmark_parallel.c src/gcn_parallel.c src/graph.c -lm -I./include/
	export OMP_NUM_THREADS=16

parallel_32:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_32 src/benchmark_parallel.c src/gcn_parallel.c src/graph.c -lm -I./include/
	export OMP_NUM_THREADS=32

parallel_64:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_64 src/benchmark_parallel.c src/gcn_parallel.c src/graph.c -lm -I./include/
	export OMP_NUM_THREADS=64

parallel_clean:
	rm -f benchmark_parallel benchmark_parallel_*

.PHONY: clean 
