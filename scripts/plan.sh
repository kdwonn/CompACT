#!/bin/bash

# Default values
NUM_NODES=1
HOST_NODE_ADDR="localhost"
CURR_NODE_RANK=0
NPROC_PER_NODE=1
PORT=0 # port 0 will automatically select an available port, it only works on single node

# Help message
function show_help {
    echo "Usage: $0 [options] [-- script_args]"
    echo "Options:"
    echo "  --nproc=N           Set number of processes per node (default: 1)"
    echo "  --port=PORT         Set rendezvous port (default: 29501)"
    echo "  --nodes=N           Set number of nodes (default: 1)"
    echo "  --host=HOSTNAME     Set host node address (default: localhost)"
    echo "  --rank=N            Set current node rank (default: 0)"
    echo "  --help              Show this help message"
    echo ""
    echo "Example: $0 --nproc=4 --port=23456 -- training.batch_size=32"
    exit 0
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc=*)
            NPROC_PER_NODE="${1#*=}"
            shift
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --nodes=*)
            NUM_NODES="${1#*=}"
            shift
            ;;
        --host=*)
            HOST_NODE_ADDR="${1#*=}"
            shift
            ;;
        --rank=*)
            CURR_NODE_RANK="${1#*=}"
            shift
            ;;
        --help)
            show_help
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Print configuration
echo "Launching torchrun with:"
echo "  Nodes: $NUM_NODES"
echo "  Host: $HOST_NODE_ADDR"
echo "  Rank: $CURR_NODE_RANK"
echo "  Processes per node: $NPROC_PER_NODE"
echo "  Rendezvous port: $PORT"
echo "  Additional arguments: $@"

# Run the command

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc-per-node=${NPROC_PER_NODE} \
  --node-rank=${CURR_NODE_RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${HOST_NODE_ADDR}:${PORT} \
  planning_eval.py $@