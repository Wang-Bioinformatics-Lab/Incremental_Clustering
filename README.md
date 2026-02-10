# Incremental Mass Spectra Clustering Workflow – User Instructions

## 1. Parameter Configuration

Before running the workflow, please ensure the following parameters are set appropriately:

- **Precursor Ion Tolerance**  
  Specify the tolerance for precursor ion matching. This can be set in **ppm** or **Da**, but the unit must be explicitly defined (e.g., `20 ppm` or `0.5 Da`).

- **Fragment Ion Tolerance**  
  Defines the tolerance for fragment ion matching.

- **Minimum Peak Intensity**  
  Set a threshold below which fragment peaks will be ignored during clustering.

- **Maximum Peak Intensity**  
  Set a threshold above which fragment peaks will be excluded to avoid noise or outliers.

- **EPS (Epsilon for DBSCAN Clustering)**  
  This parameter controls the cosine distance threshold for clustering (default: `0.1`).  
  Recommended range for cosine distance: **0.1 to 0.3**, depending on the desired clustering granularity.

---

## 2. Initial Clustering Workflow

To run the clustering workflow for a new dataset:

1. Navigate to the workflow interface.
2. Under **Input Data Folder**, select the folder containing your `mzML` spectra files.
3. Click **Submit** to start the clustering process.

---

## 3. Incremental Clustering Workflow

To cluster a new batch of spectra incrementally based on previous results:

1. Wait for the previous batch to complete. On the **Task Finished** page, click:  
   **Downstream Analysis → Downstream Analysis - Run Incremental Clustering Batch**
2. On the next page:
   - The **Input Checkpoint Folder** should automatically populate with the results from the previous batch.
   - In **Input Data Folder**, select the folder containing the new batch of spectra.
3. Click **Submit** to begin incremental clustering.

---

## 4. Viewing Clustering Results

- On the **Task Page**, click **Clustering Output List** to view clustering results organized by scan.
- To download or browse the **Consensus Spectrum File**, go to:  
  **Browse All Results**  
  The file is located at:  
  `/results/consensus.mzML`


---

## 5. Running Incremental Hyper-Spec Clustering from Terminal (Multi‑batch)

For large studies (many batches of mzML/MGF files) it is convenient to drive the incremental Hyper‑Spec workflow directly from the command line using:

`scripts/run_multi_batch_incremental.sh`

This script:

- Iterates over a sequence of batch folders (e.g. `batch_1`, `batch_2`, …) under a base directory.
- For each batch, runs the Nextflow workflow inside the Hyper‑Spec Docker image.
- Uses the **previous batch’s `results/`** as the checkpoint for the next batch, so clustering is incremental across all batches.

### 5.1. Script configuration

Open `scripts/run_multi_batch_incremental.sh` and adjust the top configuration block:

- **`BATCH_BASE_DIR`**  
  Base directory containing your batch folders. Example:
  - `.../Test_large_batch_20batches` when using mzML input, with subfolders `batch_1`, `batch_2`, …  
  - `.../Test_large_batch_20batches_mgf` when using pre‑converted MGF input.

- **`OUTPUT_BASE_DIR`**  
  Where per‑batch incremental results will be written. The script will create one subfolder per batch:
  - `OUTPUT_BASE_DIR/batch_1_results/`
  - `OUTPUT_BASE_DIR/batch_2_results/`
  - …

- **`DOCKER_IMAGE`**  
  The name/tag of the Docker image that contains Hyper‑Spec and this `Incremental_Clustering` directory.  
  This **must match** the image you build from `docker-trial/Dockerfile` (see section 6).

- **`WORKFLOW_DIR`**  
  Path inside the container where `Incremental_Clustering` is mounted.  
  The Dockerfile copies this repo to `/Incremental_Clustering`, so the default:
  - `WORKFLOW_DIR="/Incremental_Clustering"`
  is correct and usually does not need to be changed.

### 5.2. Hyper‑Spec parameters

The script exposes the key Hyper‑Spec parameters via environment variables (with defaults):

- `EPS` (DBSCAN epsilon, cosine distance), default `0.3`
- `MIN_MZ`, `MAX_MZ` (MS/MS m/z range), default `10` and `2000`
- `PRECURSOR_TOL`, `FRAGMENT_TOL` (tolerances), e.g. `20 ppm` and `0.01`

You can override them when launching the script, for example:

```bash
cd /path/to/Hyperspec-Building/Incremental_Clustering

EPS=0.6 MIN_MZ=10 MAX_MZ=2000 ./scripts/run_multi_batch_incremental.sh
```

### 5.3. Expected batch layout

`BATCH_BASE_DIR` is expected to contain subfolders named like:

```text
BATCH_BASE_DIR/
  batch_1/
    *.mzML or *.mgf
  batch_2/
    *.mzML or *.mgf
  ...
```

- If the input is mzML, the workflow converts mzML → MGF inside the container.
- If the input is MGF, the workflow uses MGF as‑is (no extra conversion).

For each batch `batch_k` the script creates:

```text
OUTPUT_BASE_DIR/batch_k_results/
  results/
    cluster_info.tsv
    cluster_summary.tsv
    consensus.mzML
    consensus.parquet
    hyperspec*_representatives.mgf
    ...
```

These `results/` folders are also used as the **checkpoint** for the next batch.

### 5.4. Running the multi‑batch script

Once `BATCH_BASE_DIR`, `OUTPUT_BASE_DIR` and `DOCKER_IMAGE` are configured:

```bash
cd /data/nas-gpu/wang/xianghu/corteva_colab_project/Hyperspec-Building/Incremental_Clustering
./scripts/run_multi_batch_incremental.sh
```

The script will:

1. Scan all `batch_*` folders under `BATCH_BASE_DIR`.
2. Skip batches whose `.../batch_k_results/results/` already exist and are non‑empty.
3. For each new batch:
   - Mount the batch folder and checkpoint (if any) into the Docker container.
   - Run `nextflow run ./nf_workflow.nf` inside the container with the configured parameters.
   - Write results to `OUTPUT_BASE_DIR/batch_k_results/`.

If any batch fails, the script stops and prints an error pointing to the log directory for that batch.

---

## 6. Building the Docker Image for Incremental Hyper‑Spec

The Docker image used by the incremental workflow is defined at the **top‑level** of this repo in:

- `docker-trial/Dockerfile`
- `docker-trial/Makefile`

The Dockerfile:

- Starts from `nvidia/cuda:12.4.0-runtime-ubuntu22.04`.
- Installs Miniforge + Mamba and creates the `hyper-spec` conda env from `docker-trial/requirements.yaml`.
- Installs Nextflow and system dependencies (libstdc++6, nvidia-utils, etc.).
- Copies:
  - `docker-trial/Hyper-Spec` → `/Hyper-Spec`
  - **`Incremental_Clustering`** → `/Incremental_Clustering`
- Builds Hyper‑Spec by running `/Hyper-Spec/install.sh`.

### 6.1. Build the image (using Makefile)

From the **`docker-trial/`** directory:

```bash
cd /data/nas-gpu/wang/xianghu/corteva_colab_project/Hyperspec-Building/docker-trial
make build-docker
```

This runs:

```bash
cd .. && docker build -f docker-trial/Dockerfile -t my-docker-image .
```

So the resulting image name is:

- `my-docker-image`

This should match the `DOCKER_IMAGE` variable in:

- `Incremental_Clustering/scripts/run_multi_batch_incremental.sh`

### 6.2. (Optional) Run interactive container for debugging

You can start an interactive GPU container with the built image:

```bash
cd /data/nas-gpu/wang/xianghu/corteva_colab_project/Hyperspec-Building/docker-trial
make run-docker-gpu
```

or equivalently:

```bash
docker run -it --gpus all my-docker-image /bin/bash
```

Inside the container:

- Hyper‑Spec code is at `/Hyper-Spec`
- Incremental workflow (this repo) is at `/Incremental_Clustering`

From there you can manually run:

```bash
cd /Incremental_Clustering
nextflow run ./nf_workflow.nf -resume -c nextflow.config --input_spectra ./data/round1
```

or use the multi‑batch script via volume mounts as described in section 5.

