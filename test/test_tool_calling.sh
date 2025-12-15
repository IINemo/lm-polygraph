#!/bin/bash
# Test script for tool calling functionality
# This script tests both optional and mandatory tool calling with Llama-3 Tulu model

set -e  # Exit on error

echo "=========================================="
echo "Testing Tool Calling Functionality"
echo "=========================================="

# Set up paths
CONFIG_DIR="./test/configs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if polygraph_eval script exists
if [ ! -f "scripts/polygraph_eval" ]; then
    echo "Error: polygraph_eval script not found at scripts/polygraph_eval"
    exit 1
fi

# Check if config files exist
if [ ! -f "$CONFIG_DIR/test_tool_calling.yaml" ]; then
    echo "Error: Test config not found at $CONFIG_DIR/test_tool_calling.yaml"
    exit 1
fi

echo ""
echo "Test 1: Optional Tool Calling"
echo "-----------------------------"
echo "Running evaluation with optional tool calling (LLM decides whether to use tool)..."
echo ""

python scripts/polygraph_eval \
    --config-dir="$CONFIG_DIR" \
    --config-name=test_tool_calling \
    cache_path=./workdir/test_output

if [ $? -eq 0 ]; then
    echo "✓ Test 1 passed: Optional tool calling completed successfully"
else
    echo "✗ Test 1 failed: Optional tool calling encountered an error"
    exit 1
fi

echo ""
echo "Test 2: Mandatory Tool Calling"
echo "------------------------------"
echo "Running evaluation with mandatory tool calling (tool must be used)..."
echo ""

python scripts/polygraph_eval \
    --config-dir="$CONFIG_DIR" \
    --config-name=test_tool_calling_mandatory \
    cache_path=./workdir/test_output

if [ $? -eq 0 ]; then
    echo "✓ Test 2 passed: Mandatory tool calling completed successfully"
else
    echo "✗ Test 2 failed: Mandatory tool calling encountered an error"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo ""
echo "Test results are saved in: ./workdir/test_output/test_tool_calling*/"
echo ""

