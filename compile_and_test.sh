#!/bin/bash

# Compile the CUDA file using nvcc
nvcc -o sha256_example cuda_sha256_example.cu

# Check if compilation was successful
echo "Checking if compilation was successful..."
if [ -f "./sha256_example" ]; then
    echo "Compilation successful. Running the test..."
    ./sha256_example
else
    echo "Compilation failed. Please ensure nvcc is installed and try again."
fi
