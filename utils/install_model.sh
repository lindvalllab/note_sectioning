#!/bin/bash

# parameters
declare -A models=(
    ["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"]="Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
    ["bartowski/gemma-2-2b-it-GGUF"]="gemma-2-2b-it-Q8_0.gguf"
    ["hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF"]="llama-3.2-3b-instruct-q8_0.gguf"
    ["hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF"]="llama-3.2-1b-instruct-q8_0.gguf"
    ["bartowski/Phi-3.5-mini-instruct-GGUF"]="Phi-3.5-mini-instruct-IQ2_M.gguf"
)

mkdir -p models
for repo in "${!models[@]}"; do
    model=${models[$repo]}
    if [ ! -f "models_custom/${model}" ]; then
        curl -L "https://huggingface.co/${repo}/resolve/main/${model}" -o "../models/${model}"
    fi
done