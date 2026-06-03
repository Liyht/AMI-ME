#!/bin/bash

# Check if a model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Available models:"
    jq -r 'keys[]' model_path.json
    exit 1
fi

MODEL_NAME=$1

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it to parse JSON."
    exit 1
fi

# Extract the model path from the JSON file
MODEL_PATH=$(jq -r --arg name "$MODEL_NAME" '.[$name]' model_path.json)

# Check if the model exists in the JSON file
if [ "$MODEL_PATH" == "null" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: Model '$MODEL_NAME' not found in model_path.json"
    echo "Available models:"
    jq -r 'keys[]' model_path.json
    exit 1
fi

if [[ "$MODEL_NAME" == "qwen3" ]]; then
   MAX_MODEL_LEN=40000
else
   MAX_MODEL_LEN=48000
fi

# Set environment variables and run the script
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Running model: $MODEL_NAME from path: $MODEL_PATH"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --dtype auto \
    --tensor_parallel_size 4 \
    --port 4060 \
    --max-model-len $MAX_MODEL_LEN
