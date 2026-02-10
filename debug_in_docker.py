#!/usr/bin/env python3
"""
Debug script to run inside Docker to check Hyper-Spec output directly.
Uses incremental_clustering's conda env for pymzml, then hyper-spec env for clustering.
"""
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def main():
    print("="*80)
    print("HYPER-SPEC DIRECT TEST IN DOCKER")
    print("="*80)
    
    # Step 1: Convert mzML to MGF using the incremental_clustering conda env
    input_mzml = "/Incremental_Clustering/data/round1/EK_Q_07.mzML"
    test_dir = "/tmp/hyperspec_debug"
    os.makedirs(test_dir, exist_ok=True)
    mgf_dir = os.path.join(test_dir, "mgf")
    os.makedirs(mgf_dir, exist_ok=True)
    mgf_path = os.path.join(mgf_dir, "EK_Q_07.mgf")
    
    print(f"\n[Step 1] Converting {input_mzml} to MGF...")
    
    # Use subprocess to call python with pymzml from the other conda env
    convert_script = f'''
import pymzml
import sys

input_mzml = "{input_mzml}"
mgf_path = "{mgf_path}"

run = pymzml.run.Reader(input_mzml, build_index_from_scratch=True)
spectrum_count = 0

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
                
                f.write("BEGIN IONS\\n")
                f.write(f"TITLE=EK_Q_07:index:{{scan}}\\n")
                f.write(f"SCANS={{scan}}\\n")
                f.write(f"PEPMASS={{pepmass}}\\n")
                f.write(f"RTINSECONDS={{rt}}\\n")
                f.write(f"CHARGE={{charge}}+\\n")
                for mz, inten in peaks:
                    f.write(f"{{mz}} {{inten}}\\n")
                f.write("END IONS\\n")
                spectrum_count += 1
            except Exception as e:
                continue

print(f"Created MGF with {{spectrum_count}} spectra")
'''
    
    # Write and run the conversion script with incremental_clustering env
    conv_script_path = "/tmp/convert_mzml.py"
    with open(conv_script_path, 'w') as f:
        f.write(convert_script)
    
    # Run with the conda env that has pymzml
    proc = subprocess.run([
        "/Incremental_Clustering/work/conda/conda_env-af7cc4fb0c245f7243e2841bb20e66c8/bin/python",
        conv_script_path
    ], capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"STDERR: {proc.stderr}")
        return
    
    # Step 2: Run Hyper-Spec directly
    print(f"\n[Step 2] Running Hyper-Spec directly...")
    output_prefix = os.path.join(test_dir, "hyperspec_direct")
    
    hyperspec_cmd = [
        "/opt/conda/envs/hyper-spec/bin/python",
        "/Hyper-Spec/src/main.py",
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
    proc = subprocess.run(hyperspec_cmd)
    print(f"Return code: {proc.returncode}")
    
    # Step 3: Check Hyper-Spec parquet output
    parquet_path = output_prefix + ".parquet"
    if os.path.exists(parquet_path):
        print(f"\n[Step 3] Checking Hyper-Spec parquet output: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        print(f"Unique clusters: {df['cluster'].nunique()}")
        print(f"Unique identifiers: {df['identifier'].nunique()}")
        
        print(f"\n--- Sample rows ---")
        print(df.head(20).to_string())
        
        # Check precursor_mz variance within clusters
        print(f"\n--- Checking precursor_mz variance ---")
        cluster_stats = df.groupby('cluster').agg({
            'precursor_mz': ['min', 'max', 'count']
        }).reset_index()
        cluster_stats.columns = ['cluster', 'mz_min', 'mz_max', 'count']
        cluster_stats['mz_range'] = cluster_stats['mz_max'] - cluster_stats['mz_min']
        
        suspicious = cluster_stats[cluster_stats['mz_range'] > 0.5]
        print(f"Clusters with mz range > 0.5 Da: {len(suspicious)} / {len(cluster_stats)}")
        
        if len(suspicious) > 0:
            print("WARNING: Hyper-Spec output has clusters with large mz range!")
            print(suspicious.head(10).to_string())
        else:
            print("Hyper-Spec output looks correct!")
        
        # Check cluster size distribution
        print(f"\n--- Cluster size distribution ---")
        size_dist = cluster_stats['count'].value_counts().sort_index()
        print(f"Size 1 (singletons): {size_dist.get(1, 0)}")
        print(f"Size 2: {size_dist.get(2, 0)}")
        print(f"Size 3-10: {sum(size_dist.get(i, 0) for i in range(3, 11))}")
        print(f"Size >10: {sum(v for k, v in size_dist.items() if k > 10)}")
        
        # Show examples of clusters with >1 member
        print(f"\n--- Examples of clusters with >1 member ---")
        multi_clusters = cluster_stats[cluster_stats['count'] > 1].head(5)['cluster'].tolist()
        for cid in multi_clusters:
            print(f"\nCluster {cid}:")
            print(df[df['cluster'] == cid][['identifier', 'scan', 'precursor_mz', 'retention_time', 'cluster']].to_string())
    else:
        print(f"ERROR: Parquet not found: {parquet_path}")
        # List what's in the test dir
        print(f"Files in {test_dir}:")
        for f in os.listdir(test_dir):
            print(f"  {f}")
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
