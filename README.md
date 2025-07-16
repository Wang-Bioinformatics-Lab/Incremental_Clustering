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

## 5.For Corteva Ometalab Workflow Deployment

First, navigate to the workflow directory:

```bash
cd /ometa/flow/workflows_user
```
Then, run the following command:

```bash
curl -L  https://github.com/Wang-Bioinformatics-Lab/Incremental_Clustering/archive/refs/heads/master.zip | bsdtar -xvf -
```
