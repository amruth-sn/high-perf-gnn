#!/bin/bash
#$ -N graphgen
#$ -cwd
#$ -j y
#$ -o output/graphgen.$TASK_ID.out
#$ -m ea
#$ -l h_rt=01:00:00
#$ -pe omp 1
#$ -t 1-10

# Define parameters
NODE_SIZES=(100000 1000000)
SEEDS_PER_SIZE=5

TASK_ID=$((SGE_TASK_ID - 1))
NODE_SIZE_INDEX=$((TASK_ID / SEEDS_PER_SIZE))
SEED=$((TASK_ID))

NODE_SIZE=${NODE_SIZES[$NODE_SIZE_INDEX]}

echo "Generating graph: nodes=${NODE_SIZE}, seed=${SEED}"

python3 scripts/generate_large_graphs.py ${NODE_SIZE} ${SEED}
