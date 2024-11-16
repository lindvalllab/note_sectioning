#!/bin/bash

# ============================
# Multi-Model Stress Test Script
# ============================

# Define the list of model paths
MODEL_PATHS=(
    "../../llama.cpp/models_custom/Phi-3.5-mini-instruct-Q6_K_L.gguf"
    "../../llama.cpp/models_custom/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "../../llama.cpp/models_custom/llama-3.2-1b-instruct-q8_0.gguf"
    "../../llama.cpp/models_custom/llama-3.2-3b-instruct-q8_0.gguf"
    "../../llama.cpp/models_custom/gemma-2-2b-it-Q6_K_L.gguf"
    # Add more model paths here as needed
)

# Define the context size and the number of runs
CONTEXT_SIZE=4096
RUNS=1000

# Path to the existing benchmarking script
BENCHMARK_SCRIPT="../src/llmbenchmark.sh"

# Check if the benchmark script exists
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "Error: Benchmark script '$BENCHMARK_SCRIPT' not found."
    exit 1
fi

# Iterate over the list of models and call the benchmark script for each model
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Running benchmark for model: $MODEL_PATH"
    
    # Call the existing benchmark script
    bash "$BENCHMARK_SCRIPT" --model "$MODEL_PATH" --context "$CONTEXT_SIZE" --runs "$RUNS"
    
    echo "Benchmark for model '$MODEL_PATH' completed."
    echo "---------------------------------------------"
done

echo "All benchmarks completed."

exit 0