#!/usr/bin/env python3
"""
Debug script to compare Hyper-Spec direct output vs workflow output.
"""
import os
import sys
import subprocess
import tempfile
import pandas as pd
from pathlib import Path

# Add bin to path
sys.path.insert(0, str(Path(__file__).parent / "bin"))

def step1_check_workflow_cluster_info():
    """Check the workflow output cluster_info.tsv for precursor_mz variance within clusters."""
    print("\n" + "="*80)
    print("STEP 1: Checking workflow cluster_info.tsv")
    print("="*80)
    
    tsv_path = "/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/incremetnal_results/results/cluster_info.tsv"
    if not os.path.exists(tsv_path):
        print(f"[ERROR] File not found: {tsv_path}")
        return
    
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique clusters: {df['cluster'].nunique()}")
    print(f"Unique filenames: {df['filename'].unique()}")
    
    # Check precursor_mz variance within each cluster
    print("\n--- Precursor m/z variance within clusters ---")
    cluster_stats = df.groupby('cluster').agg({
        'precursor_mz': ['min', 'max', 'mean', 'std', 'count']
    }).reset_index()
    cluster_stats.columns = ['cluster', 'mz_min', 'mz_max', 'mz_mean', 'mz_std', 'count']
    cluster_stats['mz_range'] = cluster_stats['mz_max'] - cluster_stats['mz_min']
    
    # Show clusters with large mz range (> 0.5 Da is suspicious, > 1 Da is definitely wrong)
    suspicious = cluster_stats[cluster_stats['mz_range'] > 0.5].sort_values('mz_range', ascending=False)
    print(f"\nClusters with precursor_mz range > 0.5 Da: {len(suspicious)}")
    print(suspicious.head(20).to_string())
    
    # Show specific examples
    if len(suspicious) > 0:
        print("\n--- Example: Cluster with largest mz range ---")
        worst_cluster = suspicious.iloc[0]['cluster']
        print(df[df['cluster'] == worst_cluster].to_string())
    
    return df


def step2_check_hyperspec_parquet():
    """Check the raw Hyper-Spec parquet output."""
    print("\n" + "="*80)
    print("STEP 2: Checking raw Hyper-Spec parquet output")
    print("="*80)
    
    # Find hyperspec parquet files in work directory
    work_dir = "/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/incremetnal_results"
    
    # First check results directory
    for fname in ["hyperspec.parquet", "hyperspec1.parquet", "hyperspec2.parquet"]:
        parquet_path = os.path.join(work_dir, "results", fname)
        if os.path.exists(parquet_path):
            print(f"\n--- Found: {parquet_path} ---")
            df = pd.read_parquet(parquet_path)
            print(f"Columns: {list(df.columns)}")
            print(f"Total rows: {len(df)}")
            print(f"Unique clusters: {df['cluster'].nunique()}")
            
            # Check precursor_mz variance
            cluster_stats = df.groupby('cluster').agg({
                'precursor_mz': ['min', 'max', 'count']
            }).reset_index()
            cluster_stats.columns = ['cluster', 'mz_min', 'mz_max', 'count']
            cluster_stats['mz_range'] = cluster_stats['mz_max'] - cluster_stats['mz_min']
            
            suspicious = cluster_stats[cluster_stats['mz_range'] > 0.5]
            print(f"Clusters with mz range > 0.5 Da in raw parquet: {len(suspicious)}")
            if len(suspicious) > 0:
                print(suspicious.head(10).to_string())
            else:
                print("(None - raw Hyper-Spec output looks correct!)")
            
            return df
    
    print("[WARNING] No hyperspec parquet found in results directory")
    return None


def step3_run_hyperspec_directly():
    """Run Hyper-Spec directly on round1 data and check output."""
    print("\n" + "="*80)
    print("STEP 3: Running Hyper-Spec directly (bypassing workflow)")
    print("="*80)
    
    # Convert mzML to MGF first
    input_mzml = "/data/nas-gpu/wang/xianghu/corteva_colab_project/Hyperspec-Building/Incremental_Clustering/data/round1/EK_Q_07.mzML"
    if not os.path.exists(input_mzml):
        print(f"[ERROR] Input file not found: {input_mzml}")
        return None
    
    # Create temp directory for this test
    test_dir = "/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/hyperspec_debug_test"
    os.makedirs(test_dir, exist_ok=True)
    mgf_dir = os.path.join(test_dir, "mgf")
    os.makedirs(mgf_dir, exist_ok=True)
    
    # Convert mzML to MGF using pymzml
    print(f"Converting {input_mzml} to MGF...")
    import pymzml
    
    mgf_path = os.path.join(mgf_dir, "EK_Q_07.mgf")
    run = pymzml.run.Reader(input_mzml, build_index_from_scratch=True)
    
    with open(mgf_path, 'w') as f:
        for spectrum in run:
            if spectrum['ms level'] == 2:
                try:
                    peaks = list(spectrum.peaks("centroided"))
                    if not peaks:
                        continue
                    
                    scan = spectrum['id']
                    pepmass = spectrum.selected_precursors[0]['mz']
                    rt = spectrum.scan_time[0]
                    charge = spectrum.selected_precursors[0].get('charge', 0) or 0
                    
                    f.write("BEGIN IONS\n")
                    f.write(f"TITLE=EK_Q_07:index:{scan}\n")
                    f.write(f"SCANS={scan}\n")
                    f.write(f"PEPMASS={pepmass}\n")
                    f.write(f"RTINSECONDS={rt}\n")
                    f.write(f"CHARGE={charge}+\n")
                    for mz, inten in peaks:
                        f.write(f"{mz} {inten}\n")
                    f.write("END IONS\n")
                except Exception as e:
                    continue
    
    print(f"Created MGF: {mgf_path}")
    
    # Run Hyper-Spec directly
    output_prefix = os.path.join(test_dir, "hyperspec_direct")
    hyperspec_cmd = [
        "/opt/conda/envs/hyper-spec/bin/python",
        "/data/nas-gpu/wang/xianghu/corteva_colab_project/Hyperspec-Building/docker-trial/Hyper-Spec/src/main.py",
        mgf_dir,
        output_prefix,
        "--cpu_core_preprocess=8",
        "--cpu_core_cluster=32",
        "--cluster_alg=dbscan",
        "--eps=0.1",
        "--min_mz_range=0",
        "--min_mz=10",
        "--max_mz=2000",
        "--cluster_charges", "2", "3",
        "--representative_mgf",
    ]
    
    print(f"Running: {' '.join(hyperspec_cmd)}")
    proc = subprocess.run(hyperspec_cmd, capture_output=True, text=True)
    print(f"Return code: {proc.returncode}")
    if proc.returncode != 0:
        print(f"STDERR: {proc.stderr}")
        return None
    
    # Check output parquet
    parquet_path = output_prefix + ".parquet"
    if os.path.exists(parquet_path):
        print(f"\n--- Direct Hyper-Spec output: {parquet_path} ---")
        df = pd.read_parquet(parquet_path)
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        print(f"Unique clusters: {df['cluster'].nunique()}")
        
        # Check precursor_mz variance
        cluster_stats = df.groupby('cluster').agg({
            'precursor_mz': ['min', 'max', 'count']
        }).reset_index()
        cluster_stats.columns = ['cluster', 'mz_min', 'mz_max', 'count']
        cluster_stats['mz_range'] = cluster_stats['mz_max'] - cluster_stats['mz_min']
        
        suspicious = cluster_stats[cluster_stats['mz_range'] > 0.5]
        print(f"\nClusters with mz range > 0.5 Da in direct run: {len(suspicious)}")
        if len(suspicious) > 0:
            print(suspicious.head(10).to_string())
        else:
            print("(None - direct Hyper-Spec output looks correct!)")
        
        return df
    else:
        print(f"[ERROR] Output parquet not found: {parquet_path}")
        return None


def step4_compare_summarize_results():
    """Compare raw parquet with summarized cluster_info.tsv to find where the problem is."""
    print("\n" + "="*80)
    print("STEP 4: Comparing raw parquet vs summarize_results_hyperspec output")
    print("="*80)
    
    # Read cluster_info.tsv
    tsv_path = "/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/incremetnal_results/results/cluster_info.tsv"
    if not os.path.exists(tsv_path):
        print(f"[ERROR] cluster_info.tsv not found")
        return
    
    df_tsv = pd.read_csv(tsv_path, sep='\t')
    
    # cluster_info.tsv is OUTPUT of incremental_clustering_hyper_spec.py::finalize_results
    # which reads from cluster_dic (built from update_cluster_dic / initial_cluster_dic)
    
    # Let's trace back: check if there's a scan_list.feather in checkpoint
    feather_path = "/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/incremetnal_results/results/scan_list.feather"
    if os.path.exists(feather_path):
        import pyarrow.feather as feather
        df_feather = feather.read_table(feather_path).to_pandas()
        print(f"\n--- scan_list.feather ---")
        print(f"Columns: {list(df_feather.columns)}")
        print(f"Total rows: {len(df_feather)}")
        print(f"Unique clusters: {df_feather['cluster_id'].nunique()}")
        
        # Check precursor_mz variance
        cluster_stats = df_feather.groupby('cluster_id').agg({
            'precursor_mz': ['min', 'max', 'count']
        }).reset_index()
        cluster_stats.columns = ['cluster_id', 'mz_min', 'mz_max', 'count']
        cluster_stats['mz_range'] = cluster_stats['mz_max'] - cluster_stats['mz_min']
        
        suspicious = cluster_stats[cluster_stats['mz_range'] > 0.5]
        print(f"\nClusters with mz range > 0.5 Da in scan_list.feather: {len(suspicious)}")
        if len(suspicious) > 0:
            print("PROBLEM IS IN scan_list.feather (before finalize_results)")
            print(suspicious.head(10).to_string())
        else:
            print("(None - scan_list.feather looks correct!)")


def step5_check_initial_cluster_dic():
    """Check initial_cluster_dic logic for cluster ID assignment."""
    print("\n" + "="*80)
    print("STEP 5: Analyzing initial_cluster_dic / update_cluster_dic logic")
    print("="*80)
    
    print("""
The issue is likely in how cluster IDs are mapped:

In initial_cluster_dic():
- Falcon/Hyper-Spec outputs cluster IDs starting from 0
- Code does: cid = int(row['cluster']) + 1  (so cluster 0 -> 1, 1 -> 2, etc.)
- This is correct.

In summarize_results_hyperspec.py:
- Reads Hyper-Spec parquet
- Maps identifier -> filename (identifier.mzML)
- Keeps original cluster IDs from Hyper-Spec

The problem might be:
1. Hyper-Spec parquet has correct clusters, but initial_cluster_dic remaps them wrongly
2. Or the identifier parsing is wrong (all become same identifier -> same file mapping)

Let me check the actual output_summary/cluster_info.tsv that summarize_results_hyperspec.py writes...
""")
    
    # Check the output_summary folder
    summary_tsv = "/data/nas-gpu/wang/xianghu/metabolomics_clustering_data/incremetnal_results/results/output_summary/cluster_info.tsv"
    if os.path.exists(summary_tsv):
        print(f"\n--- output_summary/cluster_info.tsv (from summarize_results_hyperspec) ---")
        df = pd.read_csv(summary_tsv, sep='\t')
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        print(f"Unique clusters: {df['cluster'].nunique()}")
        print(f"Sample rows:")
        print(df.head(20).to_string())
        
        # Check precursor_mz variance
        cluster_stats = df.groupby('cluster').agg({
            'precursor_mz': ['min', 'max', 'count']
        }).reset_index()
        cluster_stats.columns = ['cluster', 'mz_min', 'mz_max', 'count']
        cluster_stats['mz_range'] = cluster_stats['mz_max'] - cluster_stats['mz_min']
        
        suspicious = cluster_stats[cluster_stats['mz_range'] > 0.5]
        print(f"\nClusters with mz range > 0.5 Da in output_summary: {len(suspicious)}")
        if len(suspicious) > 0:
            print("PROBLEM IS ALREADY IN output_summary (summarize_results_hyperspec output)")
            print(suspicious.head(10).to_string())
        else:
            print("output_summary is CORRECT! Problem is in initial_cluster_dic or later.")
    else:
        print(f"[WARNING] output_summary/cluster_info.tsv not found")


if __name__ == "__main__":
    print("="*80)
    print("HYPER-SPEC CLUSTERING DEBUG SCRIPT")
    print("="*80)
    
    step1_check_workflow_cluster_info()
    step2_check_hyperspec_parquet()
    step4_compare_summarize_results()
    step5_check_initial_cluster_dic()
    
    # Skip step3 (direct run) for now - it requires hyper-spec env
    # step3_run_hyperspec_directly()
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
