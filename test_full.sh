#!/bin/bash
set -euo pipefail

# Base directory where your batches are stored
BASE_DIR="//home/user/LabData/XianghuData/Test_incremental_400"

# Path to your Nextflow script
NEXTFLOW_SCRIPT="./nf_workflow.nf"

# Loop through batches 1 to 15
for batch in {1..15}; do
    BATCH_DIR="${BASE_DIR}/batch_${batch}"
    echo "Processing ${BATCH_DIR}..."

    nextflow run "$NEXTFLOW_SCRIPT" --input_spectra "$BATCH_DIR"

    # Optionally check if the run was successful (set -e will already exit on error)
    if [ $? -ne 0 ]; then
        echo "Error encountered in batch_${batch}. Exiting."
        exit 1
    fi
done

echo "All batches processed successfully."
