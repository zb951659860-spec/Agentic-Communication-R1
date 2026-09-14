#!/bin/bash

# Generic Multi-Agent GSM8K Rollout Test Script
# Reads configuration from YAML file

# Resolve project paths dynamically
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Adjust as needed
export TOKENIZERS_PARALLELISM=false

# Configuration
CONFIG_FILE="${PROJECT_ROOT}/verl/trainer/config/ppo_trainer_latent_mas.yaml"
LOG_DIR="${REPO_ROOT}/log/GSM8K_multi_agent_test"
NUM_SAMPLES=3  # Number of samples to test

# Allow overriding config file from command line
if [ "$1" != "" ]; then
    CONFIG_FILE="$1"
fi

# Allow overriding num_samples from command line
if [ "$2" != "" ]; then
    NUM_SAMPLES="$2"
fi

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Print configuration
echo "=========================================="
echo "Config-Based Multi-Agent GSM8K Test"
echo "=========================================="
echo "Config File: ${CONFIG_FILE}"
echo "Log Directory: ${LOG_DIR}"
echo "Number of Samples: ${NUM_SAMPLES}"
echo "=========================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file does not exist: ${CONFIG_FILE}"
    exit 1
fi

# Extract some info from config file (optional, for display)
echo ""
echo "Loading configuration..."
python3 -c "
import yaml
try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    
    # Display key configurations
    multi_agent = config.get('multi_agent', {})
    env = config.get('env', {})
    
    print('Configuration Summary:')
    print(f'  Number of agents: {env.get(\"n_agents\", \"Not specified\")}')
    print(f'  Max steps: {env.get(\"max_steps\", \"Not specified\")}')
    print(f'  Action reduction: {env.get(\"action_reduction\", \"Not specified\")}')
    
    agent_models = multi_agent.get('agent_models', [])
    if agent_models:
        print('  Agent models:')
        for i, model in enumerate(agent_models[:env.get('n_agents', len(agent_models))]):
            # Shorten path for display
            if '/' in str(model):
                model_name = str(model).split('/')[-1]
            else:
                model_name = str(model)
            print(f'    Agent {i}: .../{model_name}')
except Exception as e:
    print(f'Could not parse config file: {e}')
"
echo "=========================================="

# Run the test
echo ""
echo "Starting multi-agent rollout test..."
echo "Timestamp: $(date)"
echo ""

# Use the generic config-based test script
python "${SCRIPT_DIR}/config_based_multi_agent_test.py" \
    --config "${CONFIG_FILE}" \
    --num_samples ${NUM_SAMPLES} \
    --log_dir "${LOG_DIR}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Multi-agent rollout test completed successfully!"
    echo "Logs saved to: ${LOG_DIR}"
    echo "Timestamp: $(date)"
    echo "=========================================="
    
    # Display summary if it exists
    if [ -f "${LOG_DIR}/test_summary.json" ]; then
        echo ""
        echo "Test Summary:"
        echo "----------------------------------------"
        python3 -c "
import json
try:
    with open('${LOG_DIR}/test_summary.json', 'r') as f:
        summary = json.load(f)
    print(f'Total Samples: {summary[\"total_samples\"]}')
    print(f'Correct: {summary[\"correct\"]}')
    print(f'Accuracy: {summary[\"accuracy\"]:.2%}')
    print(f'Config: {summary[\"config_file\"]}')
    
    test_config = summary.get('test_configuration', {})
    print(f'Agents Used: {test_config.get(\"n_agents\", \"Unknown\")}')
    print(f'Max Steps: {test_config.get(\"max_steps\", \"Unknown\")}')
    print(f'Action Reduction: {test_config.get(\"action_reduction\", \"Unknown\")}')
except Exception as e:
    print(f'Could not read summary: {e}')
"
        echo "----------------------------------------"
    fi
else
    echo ""
    echo "=========================================="
    echo "Multi-agent rollout test failed!"
    echo "Please check the logs in: ${LOG_DIR}"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Multi-Agent Test Complete!"
echo "=========================================="

# Optional: Display help message
echo ""
echo "Usage:"
echo "  $0 [config_file] [num_samples]"
echo ""
echo "Examples:"
echo "  $0  # Use default config and 3 samples"
echo "  $0 /path/to/custom/config.yaml  # Use custom config"
echo "  $0 /path/to/custom/config.yaml 10  # Custom config with 10 samples"
