#!/bin/bash

# Integrated Multi-Agent Test Script
# Uses MultiAgentTrajectoryCollector with config-based setup

# Resolve project paths dynamically
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# Configuration
CONFIG_FILE="${PROJECT_ROOT}/verl/trainer/config/ppo_trainer_latent_mas.yaml"
LOG_DIR="${REPO_ROOT}/log/GSM8K_integrated_test"
NUM_SAMPLES=3


# Allow command line overrides
if [ "$1" != "" ]; then
    CONFIG_FILE="$1"
fi

if [ "$2" != "" ]; then
    NUM_SAMPLES="$2"
fi

# Create log directory
mkdir -p ${LOG_DIR}

# Print configuration
echo "=========================================="
echo "Integrated Multi-Agent Test"
echo "Using MultiAgentTrajectoryCollector"
echo "=========================================="
echo "Config File: ${CONFIG_FILE}"
echo "Log Directory: ${LOG_DIR}"
echo "Number of Samples: ${NUM_SAMPLES}"
echo "=========================================="

# Check config file
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file does not exist: ${CONFIG_FILE}"
    exit 1
fi

# Display config summary
echo ""
echo "Configuration Preview:"
python3 -c "
import yaml
import sys
try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    
    env = config.get('env', {})
    multi_agent = config.get('multi_agent', {})
    
    print(f'  Max Steps: {env.get(\"max_steps\", \"?\")}')
    print(f'  Action Reduction: {env.get(\"action_reduction\", \"?\")}')
    
    models = multi_agent.get('agent_models', [])
    if models:
        print('  Models:')
        for i, m in enumerate(models[:env.get('n_agents', 2)]):
            print(f'    Agent {i}: .../{str(m).split(\"/\")[-1]}')
except Exception as e:
    print(f'  (Could not parse config: {e})')
"
echo "=========================================="

# Run the integrated test
echo ""
echo "Starting integrated multi-agent test..."
echo "Timestamp: $(date)"
echo ""

python "${SCRIPT_DIR}/integrated_multi_agent_test.py" \
    --config "${CONFIG_FILE}" \
    --num_samples ${NUM_SAMPLES} \
    --log_dir "${LOG_DIR}"

# Check status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Integrated test completed successfully!"
    echo "Logs saved to: ${LOG_DIR}"
    echo "Timestamp: $(date)"
    echo "=========================================="
    
    # Display summary
    if [ -f "${LOG_DIR}/integrated_test_summary.json" ]; then
        echo ""
        echo "Summary:"
        python3 -c "
import json
try:
    with open('${LOG_DIR}/integrated_test_summary.json', 'r') as f:
        summary = json.load(f)
    print(f'  Samples: {summary.get(\"num_samples\", \"?\")}')
    print(f'  Agents: {summary.get(\"n_agents\", \"?\")}')
    print(f'  Steps: {summary.get(\"max_steps\", \"?\")}')
except Exception as e:
    print(f'  (Could not read summary: {e})')
"
    fi
else
    echo ""
    echo "=========================================="
    echo "Integrated test failed!"
    echo "Check logs in: ${LOG_DIR}"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo ""
echo "Example:"
echo "  $0  # Use defaults"
echo "  $0 custom.yaml 5 1  # Custom config, 5 samples, agent 1"
