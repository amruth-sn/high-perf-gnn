CC = nvcc
CFLAGS = -arch compute_70 -code sm_70

SRC = main.c gcn.c graph.c
OBJ = $(SRC:.c=.o)
EXEC = gcn_program

CUDA_FLAGS = -O3  

INCLUDES = -I./

LDFLAGS =

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $(EXEC) $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(OBJ) $(EXEC) benchmark_serial benchmark_avx benchmark_parallel benchmark_parallel_*

serial:
	gcc -O2 -o benchmark_serial benchmark_serial.c graph.c gcn.c -lm

serial_clean:
	rm -f benchmark_serial

avx:
	gcc -O2 -mavx2 -o benchmark_avx benchmark_avx.c gcn_avx.c graph.c -lm

avx_clean:
	rm -f benchmark_avx

parallel:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel benchmark_parallel.c gcn_parallel.c graph.c -lm
	export OMP_NUM_THREADS=1

parallel_2:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_2 benchmark_parallel.c gcn_parallel.c graph.c -lm
	export OMP_NUM_THREADS=2

parallel_4:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_4 benchmark_parallel.c gcn_parallel.c graph.c -lm
	export OMP_NUM_THREADS=4

parallel_8:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_8 benchmark_parallel.c gcn_parallel.c graph.c -lm
	export OMP_NUM_THREADS=8

parallel_16:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_16 benchmark_parallel.c gcn_parallel.c graph.c -lm
	export OMP_NUM_THREADS=16

parallel_32:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_32 benchmark_parallel.c gcn_parallel.c graph.c -lm
	export OMP_NUM_THREADS=32

parallel_64:
	gcc -O2 -mavx2 -fopenmp -o benchmark_parallel_64 benchmark_parallel.c gcn_parallel.c graph.c -lm
	export OMP_NUM_THREADS=64

parallel_clean:
	rm -f benchmark_parallel benchmark_parallel_*

.PHONY: clean 
