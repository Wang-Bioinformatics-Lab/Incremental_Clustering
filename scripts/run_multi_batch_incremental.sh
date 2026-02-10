#!/bin/bash
# Run incremental clustering on multiple batches sequentially
# Each batch uses the previous batch's results as checkpoint
#
# Usage:
#   ./run_multi_batch_incremental.sh
#   # Or with custom parameters:
#   EPS=0.5 MIN_MZ=10 MAX_MZ=2000 ./run_multi_batch_incremental.sh

set -e  # Exit on error

# Configuration
BATCH_BASE_DIR="/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/Test_large_batch_20batches_mgf"
OUTPUT_BASE_DIR="/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/Test_large_batch_results_mgf_0.6"
DOCKER_IMAGE="hyperspec-docker-image"
WORKFLOW_DIR="/Incremental_Clustering"

# Hyper-Spec parameters (can be overridden via environment)
EPS=${EPS:-0.3}
MIN_MZ=${MIN_MZ:-10} 
MAX_MZ=${MAX_MZ:-2000}
PRECURSOR_TOL=${PRECURSOR_TOL:-"20 ppm"}
FRAGMENT_TOL=${FRAGMENT_TOL:-0.01}

# Get sorted list of batch directories
BATCH_DIRS=($(ls -d ${BATCH_BASE_DIR}/batch_* | sort -V))
NUM_BATCHES=${#BATCH_DIRS[@]}

if [ $NUM_BATCHES -eq 0 ]; then
    echo "Error: No batch directories found in ${BATCH_BASE_DIR}"
    exit 1
fi

echo "=========================================="
echo "Multi-batch Incremental Clustering"
echo "=========================================="
echo "Found ${NUM_BATCHES} batches:"
for i in "${!BATCH_DIRS[@]}"; do
    batch_name=$(basename "${BATCH_DIRS[$i]}")
    echo "  Batch $((i+1)): ${batch_name}"
done
echo ""

# Create output base directory
mkdir -p "${OUTPUT_BASE_DIR}"

# Resume: find first batch that hasn't completed clustering (skip already-done batches)
START_INDEX=0
for j in "${!BATCH_DIRS[@]}"; do
    BATCH_NAME_J=$(basename "${BATCH_DIRS[$j]}")
    BATCH_OUTPUT_J="${OUTPUT_BASE_DIR}/${BATCH_NAME_J}_results"
    if [ -d "${BATCH_OUTPUT_J}/results" ] && [ -n "$(ls -A ${BATCH_OUTPUT_J}/results 2>/dev/null)" ]; then
        echo "  Batch $((j+1)) (${BATCH_NAME_J}) already completed clustering, skipping..."
        START_INDEX=$((j+1))
    else
        break
    fi
done
if [ $START_INDEX -gt 0 ]; then
    echo "Resuming from batch $((START_INDEX+1))/${NUM_BATCHES}"
    echo ""
fi

# Track checkpoint for next batch
CHECKPOINT_DIR=""

# Process each batch sequentially (starting from START_INDEX)
for i in $(seq $START_INDEX $((NUM_BATCHES-1))); do
    BATCH_DIR="${BATCH_DIRS[$i]}"
    BATCH_NAME=$(basename "${BATCH_DIR}")
    BATCH_NUM=$((i+1))
    
    echo "=========================================="
    echo "Processing Batch ${BATCH_NUM}/${NUM_BATCHES}: ${BATCH_NAME}"
    echo "=========================================="
    
    # Output directory for this batch
    BATCH_OUTPUT="${OUTPUT_BASE_DIR}/${BATCH_NAME}_results"
    mkdir -p "${BATCH_OUTPUT}"
    
    # Checkpoint: use previous batch's results if not first batch
    if [ $i -eq 0 ]; then
        # First batch: no checkpoint
        CHECKPOINT_ARG=""
        CHECKPOINT_MOUNT=""
        echo "  Checkpoint: None (initial batch)"
    else
        # Subsequent batches: use previous batch's results as checkpoint
        PREV_BATCH_NAME=$(basename "${BATCH_DIRS[$((i-1))]}")
        PREV_OUTPUT="${OUTPUT_BASE_DIR}/${PREV_BATCH_NAME}_results"
        CHECKPOINT_DIR="${PREV_OUTPUT}/results"
        
        if [ ! -d "${CHECKPOINT_DIR}" ]; then
            echo "Error: Checkpoint directory not found: ${CHECKPOINT_DIR}"
            echo "Previous batch may have failed. Stopping."
            exit 1
        fi
        
        CHECKPOINT_ARG="${WORKFLOW_DIR}/checkpoint/results"
        CHECKPOINT_MOUNT="-v ${CHECKPOINT_DIR}:${WORKFLOW_DIR}/checkpoint/results"
        echo "  Checkpoint: ${CHECKPOINT_DIR}"
    fi
    
    echo "  Input: ${BATCH_DIR}"
    echo "  Output: ${BATCH_OUTPUT}"
    echo ""
    
    # Run workflow in Docker
    docker run --rm --gpus all \
        -v "${BATCH_DIR}":${WORKFLOW_DIR}/data/${BATCH_NAME} \
        ${CHECKPOINT_MOUNT} \
        -v "${BATCH_OUTPUT}":${WORKFLOW_DIR}/nf_output \
        "${DOCKER_IMAGE}" \
        /bin/bash -c "
            cd ${WORKFLOW_DIR} && \
            nextflow run ./nf_workflow.nf \
                -resume \
                -c nextflow.config \
                --input_spectra ./data/${BATCH_NAME} \
                ${CHECKPOINT_ARG:+--checkpoint_dir ${CHECKPOINT_ARG}} \
                --eps ${EPS} \
                --min_mz ${MIN_MZ} \
                --max_mz ${MAX_MZ} \
                --precursor_tol \"${PRECURSOR_TOL}\" \
                --fragment_tol ${FRAGMENT_TOL}
        "
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "Error: Batch ${BATCH_NUM} (${BATCH_NAME}) failed!"
        echo "Stopping workflow. Check logs in ${BATCH_OUTPUT}"
        exit 1
    fi
    
    echo ""
    echo "✓ Batch ${BATCH_NUM} (${BATCH_NAME}) completed successfully"
    echo "  Results saved to: ${BATCH_OUTPUT}"
    echo ""
done

echo "=========================================="
echo "All batches completed successfully!"
echo "Final results: ${OUTPUT_BASE_DIR}"
echo "=========================================="
