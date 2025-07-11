#!/bin/bash

# Download script for EK_Q_07.mzML and EK_Q_07_1.mzML from fileserver.wanglab.science
# This script downloads the files and verifies their integrity

set -e  # Exit on any error

# Configuration
BASE_URL="https://fileserver.wanglab.science/xianghu"
FILES=("EK_Q_07.mzML" "EK_Q_07_1.mzML")
DOWNLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$DOWNLOAD_DIR/download.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if file exists and is valid
check_file() {
    local file="$1"
    local file_path="$DOWNLOAD_DIR/$file"
    
    if [[ ! -f "$file_path" ]]; then
        error "File $file does not exist"
        return 1
    fi
    
    # Check file size (should be > 0)
    local file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null)
    if [[ $file_size -eq 0 ]]; then
        error "File $file is empty"
        return 1
    fi
    
    # Check if it's a valid mzML file by looking for mzML header
    if ! grep -q "<?xml" "$file_path" 2>/dev/null; then
        error "File $file does not appear to be a valid XML file"
        return 1
    fi
    
    if ! grep -q "mzML" "$file_path" 2>/dev/null; then
        error "File $file does not appear to be a valid mzML file"
        return 1
    fi
    
    success "File $file is valid (size: ${file_size} bytes)"
    return 0
}

# Function to download a file
download_file() {
    local file="$1"
    local url="$BASE_URL/$file"
    local file_path="$DOWNLOAD_DIR/$file"
    
    log "Downloading $file from $url..."
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        error "curl is not installed. Please install curl first."
        return 1
    fi
    
    # Download with progress bar and resume capability
    if curl -L -C - --progress-bar -o "$file_path" "$url"; then
        success "Downloaded $file successfully"
        return 0
    else
        error "Failed to download $file"
        return 1
    fi
}

# Function to verify mzML file structure
verify_mzml() {
    local file="$1"
    local file_path="$DOWNLOAD_DIR/$file"
    
    log "Verifying mzML structure for $file..."
    
    # Check for required mzML elements
    local checks=(
        "mzML"
        "run"
        "spectrumList"
        "spectrum"
    )
    
    for check in "${checks[@]}"; do
        if ! grep -q "$check" "$file_path" 2>/dev/null; then
            warning "Element '$check' not found in $file"
        else
            log "Found element '$check' in $file"
        fi
    done
    
    # Count spectra
    local spectrum_count=$(grep -c "<spectrum" "$file_path" 2>/dev/null || echo "0")
    log "Found $spectrum_count spectra in $file"
    
    if [[ $spectrum_count -gt 0 ]]; then
        success "mzML file $file appears to be valid with $spectrum_count spectra"
        return 0
    else
        error "No spectra found in $file"
        return 1
    fi
}

# Main execution
main() {
    log "Starting download process..."
    log "Download directory: $DOWNLOAD_DIR"
    log "Base URL: $BASE_URL"
    
    # Create download directory if it doesn't exist
    mkdir -p "$DOWNLOAD_DIR"
    
    # Check if files already exist
    local all_files_exist=true
    for file in "${FILES[@]}"; do
        if [[ ! -f "$DOWNLOAD_DIR/$file" ]]; then
            all_files_exist=false
            break
        fi
    done
    
    if $all_files_exist; then
        log "All files already exist. Checking integrity..."
    fi
    
    # Download and verify each file
    local success_count=0
    for file in "${FILES[@]}"; do
        echo ""
        log "Processing $file..."
        
        # Download if not exists or if forced
        if [[ ! -f "$DOWNLOAD_DIR/$file" ]] || [[ "$1" == "--force" ]]; then
            if download_file "$file"; then
                success_count=$((success_count + 1))
            else
                continue
            fi
        fi
        
        # Check file integrity
        if check_file "$file"; then
            # Verify mzML structure
            if verify_mzml "$file"; then
                success_count=$((success_count + 1))
            fi
        fi
    done
    
    echo ""
    log "Download process completed."
    
    # Summary
    local total_files=${#FILES[@]}
    local expected_success=$((total_files * 2))  # download + verify for each file
    
    if [[ $success_count -eq $expected_success ]]; then
        success "All files downloaded and verified successfully!"
        echo ""
        echo "Files available in: $DOWNLOAD_DIR"
        ls -lh "$DOWNLOAD_DIR"/*.mzML 2>/dev/null || true
    else
        error "Some files failed to download or verify. Check the log: $LOG_FILE"
        exit 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [--force]"
    echo ""
    echo "Downloads EK_Q_07.mzML and EK_Q_07_1.mzML from fileserver.wanglab.science"
    echo ""
    echo "Options:"
    echo "  --force    Force re-download even if files exist"
    echo "  --help     Show this help message"
    echo ""
    echo "Files will be downloaded to: $DOWNLOAD_DIR"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        usage
        exit 0
        ;;
    --force)
        main --force
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        usage
        exit 1
        ;;
esac 