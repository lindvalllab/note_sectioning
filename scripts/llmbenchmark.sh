#!/bin/bash

# ============================
# Stress Test Script for LLaMA Server
# ============================

# Function to display usage information
usage() {
    echo "Usage: $0 --model <path_to_model> --context <context_size> --runs <number_of_runs>"
    echo ""
    echo "Options:"
    echo "  --model <path>       Path to the LLaMA model file (e.g., ../models/Phi-3.5-mini-instruct-Q6_K_L.gguf) [required]"
    echo "  --context <size>     Context size (-c flag when running the server) [required]"
    echo "  --runs <number>      Number of HTTP POST requests to make for benchmarking [required]"
    echo ""
    echo "Example:"
    echo "  ./llmbenchmark.sh --model ../models/Phi-3.5-mini-instruct-Q6_K_L.gguf --context 4096 --runs 100"
    exit 1
}

# ----------------------------
# Parse Command-Line Arguments
# ----------------------------

# Initialize variables
MODEL_PATH=""
CONTEXT_SIZE=""
RUNS=""

# Function to parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            --model)
                MODEL_PATH="$2"
                shift 2
                ;;
            --context)
                CONTEXT_SIZE="$2"
                shift 2
                ;;
            --runs)
                RUNS="$2"
                shift 2
                ;;
            *)
                echo "Error: Unknown option '$1'"
                usage
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$MODEL_PATH" || -z "$CONTEXT_SIZE" || -z "$RUNS" ]]; then
        echo "Error: Missing required arguments."
        usage
    fi
}

# Call the argument parsing function
parse_args "$@"

# ----------------------------
# Setup Directories and Log Files
# ----------------------------

# Determine the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the output directory
OUTPUT_DIR="$SCRIPT_DIR/../outputs"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Determine the next benchmark log file name (benchmark_0.log, benchmark_1.log, etc.)
get_next_benchmark_file() {
    local log_dir="$1"
    local base_name="benchmark_"
    local ext=".log"
    local index=0

    while [[ -e "${log_dir}/${base_name}${index}${ext}" ]]; do
        ((index++))
    done

    echo "${log_dir}/${base_name}${index}${ext}"
}

BENCHMARK_FILE=$(get_next_benchmark_file "$OUTPUT_DIR")

# ----------------------------
# Start the LLaMA Server
# ----------------------------

# Define the path to the llama executable
LLAMA_EXEC="$SCRIPT_DIR/../../llama.cpp/build/bin/llama-server"

# Check if llama executable exists and is executable
if [ ! -x "$LLAMA_EXEC" ]; then
    echo "Error: 'llama-server' executable not found or not executable at '$LLAMA_EXEC'." | tee -a "$BENCHMARK_FILE"
    exit 1
fi

# Define the server log file
SERVER_LOG="$OUTPUT_DIR/server.log" 

# Start the server in the background
echo "Starting llama-server..."
"$LLAMA_EXEC" -m "$MODEL_PATH" -c "$CONTEXT_SIZE" -ngl 1000 --temp 0 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
# echo "Server PID: $SERVER_PID" | tee -a "$BENCHMARK_FILE"

echo "Model Name: $MODEL_PATH" | tee -a "$BENCHMARK_FILE"

# ----------------------------
# Monitor Server Startup
# ----------------------------

# Initialize variables to capture timestamps
FIRST_TIMESTAMP=""
LAST_TIMESTAMP=""

# echo "Waiting for server to start..."

# Function to extract timestamp from a log line
extract_timestamp() {
    echo "$1" | grep -oP 'timestamp=\K\d+'
}

# Loop until both timestamps are captured
while true; do
    # Check if the server log contains 'build info' for the first timestamp
    if [[ -z "$FIRST_TIMESTAMP" ]]; then
        FIRST_LINE=$(grep 'build info' "$SERVER_LOG" | head -n 1)
        if [[ -n "$FIRST_LINE" ]]; then
            FIRST_TIMESTAMP=$(extract_timestamp "$FIRST_LINE")
        fi
    fi

    # Check if the server log contains 'all slots are idle' for the last timestamp
    LAST_LINE=$(grep 'all slots are idle' "$SERVER_LOG" | tail -n 1)
    if [[ -n "$LAST_LINE" ]]; then
        LAST_TIMESTAMP=$(extract_timestamp "$LAST_LINE")
    fi

    # If both timestamps are captured, break the loop
    if [[ -n "$FIRST_TIMESTAMP" && -n "$LAST_TIMESTAMP" ]]; then
        break
    fi

    # Sleep briefly before checking again
    sleep 0.1
done

# Compute server startup time
STARTUP_TIME=$((LAST_TIMESTAMP - FIRST_TIMESTAMP))
echo "Server startup time (s): $STARTUP_TIME" | tee -a "$BENCHMARK_FILE"

# ----------------------------
# Perform HTTP POST Requests
# ----------------------------

# Use Python to generate a random prompt with -c tokens
PROMPT=$(python3 ../utils/generate_prompt.py "$CONTEXT_SIZE")
# Log the generated prompt
# echo "Generated Prompt (token-based, $CONTEXT_SIZE tokens): $PROMPT" | tee -a "$BENCHMARK_FILE"
echo "Generated Prompt (token-based, $CONTEXT_SIZE tokens)" | tee -a "$BENCHMARK_FILE"

N_PREDICT=50

# Initialize counters
# Initialize aggregate metrics
TOTAL_PROMPT_MS=0
TOTAL_PREDICTED_MS=0
TOTAL_TOKENS_PREDICTED=0
TOTAL_TOKENS_EVALUATED=0
TOTAL_TOKENS_CACHED=0

echo "Running $RUNS requests..." | tee -a "$BENCHMARK_FILE"

for ((i=1; i<=RUNS; i++)); do
    # echo "Run $i:" | tee -a "$BENCHMARK_FILE"
    
    # Capture the full JSON response
    RESPONSE=$(curl -s --request POST \
        --url http://localhost:8080/completion \
        --header "Content-Type: application/json" \
        --data "{\"prompt\": \"${PROMPT}\", \"n_predict\": ${N_PREDICT}}")
    
    # Check if the request was successful
    if [[ $? -eq 0 ]]; then
        # Extract metrics using jq
        PROMPT_MS=$(echo "$RESPONSE" | jq '.timings.prompt_ms')
        PREDICTED_MS=$(echo "$RESPONSE" | jq '.timings.predicted_ms')
        TOKENS_PREDICTED=$(echo "$RESPONSE" | jq '.tokens_predicted')
        TOKENS_EVALUATED=$(echo "$RESPONSE" | jq '.tokens_evaluated')
        TOKENS_CACHED=$(echo "$RESPONSE" | jq '.tokens_cached')
        
        # # Log the extracted metrics
        # echo "  Success:" | tee -a "$BENCHMARK_FILE"
        # echo "    Prompt Time (ms): $PROMPT_MS" | tee -a "$BENCHMARK_FILE"
        # echo "    Predicted Time (ms): $PREDICTED_MS" | tee -a "$BENCHMARK_FILE"
        # echo "    Tokens Predicted: $TOKENS_PREDICTED" | tee -a "$BENCHMARK_FILE"
        # echo "    Tokens Evaluated: $TOKENS_EVALUATED" | tee -a "$BENCHMARK_FILE"
        # echo "    Tokens Cached: $TOKENS_CACHED" | tee -a "$BENCHMARK_FILE"
        
        # Accumulate the metrics
        TOTAL_PROMPT_MS=$(awk "BEGIN {print $TOTAL_PROMPT_MS + $PROMPT_MS}")
        TOTAL_PREDICTED_MS=$(awk "BEGIN {print $TOTAL_PREDICTED_MS + $PREDICTED_MS}")
        TOTAL_TOKENS_PREDICTED=$((TOTAL_TOKENS_PREDICTED + TOKENS_PREDICTED))
        TOTAL_TOKENS_EVALUATED=$((TOTAL_TOKENS_EVALUATED + TOKENS_EVALUATED))
        TOTAL_TOKENS_CACHED=$((TOTAL_TOKENS_CACHED + TOKENS_CACHED))
        
        ((SUCCESS_COUNT++))
    else
        # echo "  Failed to get a response." | tee -a "$BENCHMARK_FILE"
        ((FAIL_COUNT++))
    fi
done

# ----------------------------
# Calculate and Log Aggregated Metrics
# ----------------------------

echo "Benchmarking Metrics:" | tee -a "$BENCHMARK_FILE"
echo "----------------------------------" | tee -a "$BENCHMARK_FILE"
# echo "Total Runs: $RUNS" | tee -a "$BENCHMARK_FILE"
# echo "Successful Runs: $SUCCESS_COUNT" | tee -a "$BENCHMARK_FILE"
# echo "Failed Runs: $FAIL_COUNT" | tee -a "$BENCHMARK_FILE"

if [[ $SUCCESS_COUNT -gt 0 ]]; then
    AVG_PROMPT_MS=$(awk "BEGIN {print $TOTAL_PROMPT_MS / $SUCCESS_COUNT}")
    AVG_PREDICTED_MS=$(awk "BEGIN {print $TOTAL_PREDICTED_MS / $SUCCESS_COUNT}")
    AVG_TOKENS_PREDICTED=$(awk "BEGIN {print $TOTAL_TOKENS_PREDICTED / $SUCCESS_COUNT}")
    AVG_TOKENS_EVALUATED=$(awk "BEGIN {print $TOTAL_TOKENS_EVALUATED / $SUCCESS_COUNT}")
    AVG_TOKENS_CACHED=$(awk "BEGIN {print $TOTAL_TOKENS_CACHED / $SUCCESS_COUNT}")
    TOTAL_PROMPT_SEC=$(awk "BEGIN {print $TOTAL_PROMPT_MS / 1000}")
    PROMPT_PROCESSING_SPEED=$(awk "BEGIN {print $TOTAL_TOKENS_EVALUATED / $TOTAL_PROMPT_SEC}")
    TOTAL_PREDICTED_SEC=$(awk "BEGIN {print $TOTAL_PREDICTED_MS / 1000}")
    PROMPT_GENERATION_SPEED=$(awk "BEGIN {print $TOTAL_TOKENS_PREDICTED / $TOTAL_PREDICTED_SEC}")
    
    echo "Prompt Processing Speed (tokens/sec): $PROMPT_PROCESSING_SPEED" | tee -a "$BENCHMARK_FILE"
    echo "Prompt Generation Speed (tokens/sec): $PROMPT_GENERATION_SPEED" | tee -a "$BENCHMARK_FILE"

else
    echo "No successful runs to calculate average metrics." | tee -a "$BENCHMARK_FILE"
fi

# ----------------------------
# Stop the LLaMA Server
# ----------------------------

# echo "Stopping LLaMA server (PID: $SERVER_PID)..." | tee -a "$BENCHMARK_FILE"
kill "$SERVER_PID"

# Wait for the server process to terminate
wait "$SERVER_PID" 2>/dev/null

# echo "Server stopped." | tee -a "$BENCHMARK_FILE"

# ----------------------------
# Cleanup
# ----------------------------

# Remove the temporary server log file
rm "$SERVER_LOG"

# echo "Benchmarking completed. Results saved to '$BENCHMARK_FILE'." | tee -a "$BENCHMARK_FILE"

exit 0