# Compiler and flags
CC = nvcc
CFLAGS = -arch compute_70 -code sm_70

# File names
SRC = main.c gcn.c graph.c
OBJ = $(SRC:.c=.o)
EXEC = gcn_program

# CUDA flags
CUDA_FLAGS = -O3  # Optimization flag for CUDA

# Include directories
INCLUDES = -I./

# Linker flags
LDFLAGS =

# Rule to build the program
$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $(EXEC) $(LDFLAGS)

# Rule to compile .c files to .o
%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

# Clean rule to remove object files and executable
clean:
	rm -f $(OBJ) $(EXEC)

.PHONY: clean
