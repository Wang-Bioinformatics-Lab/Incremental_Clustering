#!/usr/bin/env python3
import time
import os
import sys
import argparse
import subprocess
import pandas as pd
import shutil
import csv
from pathlib import Path
import pymzml
from tqdm import tqdm
import pyopenms as oms
import random
from multiprocessing import Pool
import psutil
import pickle
import datetime
import h5py
import numpy as np
from csv import DictWriter
import pyarrow.parquet as pq
import pyarrow.feather as feather
from pymzml.obo import OboTranslator
from numcodecs import VLenArray, Blosc  # Add these imports at the top
import numcodecs
import pyarrow as pa
from collections import defaultdict
import hashlib
from joblib import Parallel, delayed
import uuid
import concurrent.futures
import io
from functools import lru_cache
from functools import partial
import gc


##############################################################################
# 1) FALCON + SUMMARIZE + MZML READ/WRITE FUNCTIONS
##############################################################################

def run_falcon(mzml_pattern, output_prefix="falcon", precursor_tol="20 ppm", fragment_tol=0.05,
               min_mz_range=0, min_mz=0, max_mz=30000, eps=0.1):
    """
    Runs Falcon with specified parameters.
    """
    command = (
        f"falcon {mzml_pattern} {output_prefix} "
        f"--export_representatives "
        f"--precursor_tol {precursor_tol} "
        f"--fragment_tol {fragment_tol} "
        f"--min_mz_range {min_mz_range} "
        f"--min_mz {min_mz} --max_mz {max_mz} "
        f"--eps {eps} "
        f"--hash_len 400 "
        f"--n_neighbors_ann 64 "
        f"--n_probe 16 "
        f"--batch_size 32768 "
    )
    print(f"[run_falcon] Running: {command}")
    process = subprocess.Popen(command, shell=True)
    retcode = process.wait()
    # if retcode != 0:
    #     raise RuntimeError(f"Falcon failed with exit code {retcode}")


def summarize_output(output_path, summarize_script="summarize_results.py", falcon_csv="falcon.csv"):
    """
    Summarizes falcon.csv into a 'cluster_info.tsv', stored in a subdir 'output_summary'.
    """
    output_dir = os.path.join(output_path, "output_summary")
    os.makedirs(output_dir, exist_ok=True)
    python_cmd = sys.executable  # or "python3"
    command = f"{python_cmd} {summarize_script} {falcon_csv} {output_dir}"
    print(f"[summarize_output] Running: {command}")
    proc = subprocess.Popen(command, shell=True)
    proc.wait()
    return os.path.join(output_dir, "cluster_info.tsv")


def get_original_file_path(filename, original_filepah):
    return os.path.join(original_filepah, filename)


def read_mzml(filepath):
    """
    Reads MS2 spectra from an mzML file via pymzml, returns list of spectrum dicts.
    """
    spectra = []
    run = pymzml.run.Reader(filepath, build_index_from_scratch=True)
    for spectrum in run:
        if spectrum['ms level'] == 2:
            spectrum_dict = {
                'peaks': [],
                # 'm/z array': [],
                # 'intensity array': [],
                'precursor_mz': spectrum.selected_precursors[0]['mz'],
                'rtinseconds': spectrum.scan_time[0],
                'scans': spectrum['id'],
                'charge': spectrum.selected_precursors[0].get('charge', None)
            }
            for mz, intensity in spectrum.peaks("centroided"):
                spectrum_dict['peaks'].append((mz, intensity))
                # spectrum_dict['m/z array'].append(mz)
                # spectrum_dict['intensity array'].append(intensity)
            spectra.append(spectrum_dict)
    return spectra


def read_mzml_parallel(folder_path, max_workers=8):
    """
    Reads all *.mzML in a folder (except 'consensus.mzML') in parallel,
    returning a dict ( (basename_no_ext, scan_id) -> spectrum_data ).
    """
    from multiprocessing import Pool
    spectra_dict = {}
    mzml_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".mzML") and f != "consensus.mzML"
    ]
    with Pool(processes=max_workers) as pool:
        for partial_dict in pool.map(_read_mzml_file, mzml_files):
            spectra_dict.update(partial_dict)
    return spectra_dict


def _read_mzml_file(filepath):
    """
    Helper for parallel read_mzml_parallel
    """
    out = {}
    try:
        specs = read_mzml(filepath)
        base = os.path.splitext(os.path.basename(filepath))[0]
        for s in specs:
            out[(base, s['scans'])] = s
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return out


def _custom_merge_mzml_files_parallel(file_list, output_path, threads=8):
    """
    Simple serial streaming mzML merger for debugging.
    This approach preserves original IDs using minimal memory.
    """
    if not file_list:
        print("No files to merge")
        return

    if len(file_list) == 1:
        # Single file, just copy it
        shutil.copy2(file_list[0], output_path)
        return

    print(f"Serial streaming merge of {len(file_list)} mzML files into {output_path}")

    # Use true streaming merge directly
    _true_streaming_merge(file_list, output_path)


def _true_streaming_merge(file_list, output_path):
    """
    Simple serial streaming merge that processes files one by one.
    """
    if not file_list:
        return

    if len(file_list) == 1:
        shutil.copy2(file_list[0], output_path)
        return

    # Use OpenMS consumer for true streaming output
    output_consumer = oms.PlainMSDataWritingConsumer(str(output_path))
    total_spectra = 0

    # Process each file individually with progress bar
    for file_idx, file_path in enumerate(tqdm(file_list, desc="Merging files", unit="file")):
        try:
            # Use pymzml to read file streamingly
            run = pymzml.run.Reader(file_path, build_index_from_scratch=True)

            for spectrum in run:
                try:
                    if spectrum['ms level'] == 2:  # Only MS2 spectra
                        # Convert spectrum to OpenMS format
                        oms_spec = oms.MSSpectrum()
                        oms_spec.setMSLevel(2)

                        # Keep the preserved ID
                        oms_spec.setNativeID(spectrum['id'])

                        # Set retention time
                        if spectrum.scan_time:
                            rt, unit = spectrum.scan_time
                            oms_spec.setRT(rt * 60 if unit == 'minute' else rt)

                        # Set precursor information
                        if spectrum.selected_precursors:
                            precursor = oms.Precursor()
                            precursor.setMZ(spectrum.selected_precursors[0]['mz'])
                            precursor.setCharge(spectrum.selected_precursors[0].get('charge', 1))
                            oms_spec.setPrecursors([precursor])

                        # Set peaks with robust handling
                        peaks = spectrum.peaks("centroided")
                        if peaks is not None and len(peaks) > 0:
                            clean_peaks = []
                            for peak in peaks:
                                if len(peak) == 2:
                                    try:
                                        mz_val, int_val = peak[0], peak[1]
                                        # Ensure we have scalar values
                                        if hasattr(mz_val, '__iter__') and not isinstance(mz_val, str):
                                            mz_val = mz_val[0] if len(mz_val) > 0 else 0.0
                                        if hasattr(int_val, '__iter__') and not isinstance(int_val, str):
                                            int_val = int_val[0] if len(int_val) > 0 else 0.0

                                        mz_f = float(mz_val)
                                        int_f = float(int_val)
                                        if np.isfinite(mz_f) and np.isfinite(int_f):
                                            clean_peaks.append((mz_f, int_f))
                                    except Exception as e:
                                        continue

                            if len(clean_peaks) > 0:
                                peaks_array = np.array(clean_peaks, dtype=np.float64)
                                mz = peaks_array[:, 0]
                                intensity = peaks_array[:, 1]
                                oms_spec.set_peaks((mz, intensity))

                        # Stream the spectrum directly to output
                        output_consumer.consumeSpectrum(oms_spec)
                        total_spectra += 1

                except Exception as e:
                    # Skip problematic spectra but continue processing the file
                    continue

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue

    # PlainMSDataWritingConsumer doesn't have a close() method
    # The file will be finalized when the consumer goes out of scope
    # We can get the final count from the consumer
    total_spectra = output_consumer.getNrSpectraWritten()

    print(f"Successfully merged {total_spectra} spectra into {output_path}")


def _batch_merge_mzml_files(file_list, output_path, threads=40, batch_size=100):
    """
    Simplified merge that directly uses serial streaming merge.
    """
    # Simply call the serial streaming merge
    _custom_merge_mzml_files_parallel(file_list, output_path, threads)


def write_singletons_mzml(scan_list, output_path):
    """
    Write an mzML file from a list of (fp, sc) with scan IDs as {filename}_{scan}.
    This version uses parallel file fetching with serial streaming merge.
    """
    parent_dir = Path(output_path).parent
    temp_root = parent_dir / "temp_singletons"
    temp_root.mkdir(exist_ok=True)
    fetch_write = defaultdict(list)
    path_map = {}

    # Group scans by file
    for fp, sc in scan_list:
        path_hash = hashlib.md5(str(fp).encode()).hexdigest()[:8]
        base_name = Path(fp).stem
        path_key = f"{path_hash}_{base_name}"
        path_map[path_key] = fp
        new_id = f"{path_key}_{sc}"
        fetch_write[fp].append((sc, new_id))

    # Save path map for reverse mapping
    map_file = Path(output_path).with_suffix(".pathmap.feather")
    path_df = pd.DataFrame.from_dict(path_map, orient='index', columns=["full_path"])
    feather.write_feather(path_df, map_file)

    # Prepare tasks for parallel processing
    tasks = [(fp, scans) for fp, scans in fetch_write.items() if Path(fp).exists()]
    uid = uuid.uuid4().hex[:6]
    per_proc_temp_dir = temp_root / f"joblib_{uid}"
    per_proc_temp_dir.mkdir(parents=True, exist_ok=True)

    # Use at least half of available cores for processing
    available_cores = os.cpu_count() or 1
    N_JOBS = max(40, min(available_cores // 2, 96))  # At least 40, up to half of cores, max 96

    def safe_process(fp, scans):
        return _process_fetch_file(fp, scans, per_proc_temp_dir)

    print(f"Starting to process {len(tasks)} files with {N_JOBS} parallel jobs...")
    # Process all tasks in parallel with dynamic scheduling
    all_temp_files = Parallel(n_jobs=N_JOBS)(
        delayed(safe_process)(fp, scans) for fp, scans in tasks
    )
    # Filter out None results from failed tasks
    all_temp_files = list(filter(None, all_temp_files))
    gc.collect()

    # --- Serial Streaming Merging ---
    final_size = 0
    if all_temp_files:
        print(f"Serial streaming merge of {len(all_temp_files)} temporary mzML files into {output_path}")
        try:
            # Use the serial streaming merger
            _batch_merge_mzml_files(all_temp_files, output_path, threads=N_JOBS, batch_size=50)
            final_size = len(all_temp_files)
            print("Merging complete.")
        except Exception as e:
            print(f"[Error] Failed during final merge: {e}")
        finally:
            for temp_file in all_temp_files:
                try:
                    if temp_file and Path(temp_file).exists():
                        Path(temp_file).unlink()
                except OSError as e:
                    print(f"[Warning] Could not delete temp file {temp_file}: {e}")
    else:
        print("No singleton spectra were found to write.")
        oms.MzMLFile().store(output_path, oms.MSExperiment())

    try:
        shutil.rmtree(per_proc_temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"[Warning] Failed to delete temp dir {per_proc_temp_dir}: {e}")

    return final_size


def _write_ref_spectra_batch(spectra_batch, temp_dir):
    """
    Writes a batch of in-memory spectra to a unique temporary mzML file.
    spectra_batch: A list of (scan_id, spectrum_data) tuples.
    """
    try:
        exp = oms.MSExperiment()
        for cid, sp_data in spectra_batch:
            _add_spectrum(exp, sp_data, cid)

        if exp.size() > 0:
            # Create a unique filename to avoid collisions in parallel environments
            unique_id = uuid.uuid4().hex[:8]
            output_path = temp_dir / f"ref_spectra_batch_{unique_id}.mzML"
            oms.MzMLFile().store(str(output_path), exp)
            return str(output_path)
    except Exception as e:
        print(f"[Error] Failed to write reference spectra batch: {e}")
    return None


def write_mzml(cluster_dic, out_path):
    """
    Writes a consensus mzML file from a cluster dictionary.
    This version uses parallel file fetching with serial streaming merge.
    """
    ref_write_tasks = []  # List of (cid, spectrum_data) for in-memory spectra
    fetch_write = defaultdict(list)  # filepath:[(scan, new_id), ...]

    # 1. Categorize all spectra first. This is a fast, in-memory operation.
    for cid, data in cluster_dic.items():
        scan_list = data.get('scan_list', [])
        n_pool = len(scan_list)

        if 'spectrum' in data:
            # Clusters with reps: singletons (n=1), small (2-5), large (>5)
            # We write the representative for any cluster that has one and is not a pure singleton to be fetched.
            if n_pool > 1:
                ref_write_tasks.append((cid, data['spectrum']))

        if n_pool == 1 and scan_list:
            # Singletons are fetched from original files
            fp, sc = scan_list[0]
            fetch_write[fp].append((sc, cid))
        elif n_pool > 5:
            # For large clusters, fetch 5 random samples in addition to the representative
            chosen = random.sample(scan_list, 5)
            for i, (fp, sc) in enumerate(chosen):
                new_id = f"{cid}_{i + 1}"
                fetch_write[fp].append((sc, new_id))

    # --- Start of Parallel Writing Process with Serial Merge ---
    temp_dir = Path(out_path).parent / "temp_consensus"
    temp_dir.mkdir(exist_ok=True)

    all_temp_files = []
    N_JOBS = min(40, os.cpu_count() or 1)

    # 2. Parallelize writing of in-memory 'ref' spectra
    if ref_write_tasks:
        # Use at least half of available cores
        available_cores = os.cpu_count() or 1
        N_JOBS = max(40, min(available_cores // 2, 96))

        print(f"Writing {len(ref_write_tasks)} in-memory reference spectra in parallel using {N_JOBS} jobs...")
        # Chunk tasks for distribution to workers
        chunk_size = (len(ref_write_tasks) + N_JOBS - 1) // N_JOBS
        ref_chunks = [ref_write_tasks[i:i + chunk_size] for i in range(0, len(ref_write_tasks), chunk_size)]

        ref_temp_files = Parallel(n_jobs=N_JOBS)(
            delayed(_write_ref_spectra_batch)(chunk, temp_dir) for chunk in ref_chunks
        )
        all_temp_files.extend(filter(None, ref_temp_files))
        del ref_write_tasks, ref_chunks, ref_temp_files
        gc.collect()

    # 3. Parallelize fetching remaining spectra from disk
    if fetch_write:
        fetch_tasks = [(fp, scans) for fp, scans in fetch_write.items() if Path(fp).exists()]

        def safe_process(fp, scans):
            return _process_fetch_file(fp, scans, temp_dir)

        # Use at least half of available cores
        available_cores = os.cpu_count() or 1
        N_JOBS = max(40, min(available_cores // 2, 96))

        print(f"Fetching spectra from {len(fetch_tasks)} source files in parallel using {N_JOBS} jobs...")

        fetched_temp_files = Parallel(n_jobs=N_JOBS)(
            delayed(safe_process)(fp, scans) for fp, scans in fetch_tasks
        )
        all_temp_files.extend(filter(None, fetched_temp_files))

    # 4. Serial streaming merge all temporary files into the final output
    num_spectra_written = 0  # Initialize variable
    if all_temp_files:
        print(f"Serial streaming merge of {len(all_temp_files)} temporary mzML files into {out_path}")
        try:
            # Use the serial streaming merger
            _batch_merge_mzml_files(all_temp_files, out_path, threads=N_JOBS, batch_size=50)
            # To get an accurate count, we must load the final file header.
            # This is faster than loading the whole file.
            final_exp = oms.MSExperiment()
            oms.MzMLFile().load(str(out_path), final_exp)
            num_spectra_written = final_exp.size()

        except Exception as e:
            print(f"[Error] Failed during final merge: {e}")
        finally:
            # Clean up all temporary files
            for temp_file in all_temp_files:
                try:
                    if temp_file and Path(temp_file).exists():
                        Path(temp_file).unlink()
                except OSError as e:
                    print(f"[Warning] Could not delete temp file {temp_file}: {e}")
    else:
        print("No consensus spectra were found to write.")
        oms.MzMLFile().store(out_path, oms.MSExperiment())  # Create empty file

    # 5. Final cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    return num_spectra_written


def _process_fetch_file(file_path, scan_pairs, temp_dir):
    """Process a single file with enhanced error handling"""
    try:
        exp = oms.MSExperiment()
        run = pymzml.run.Reader(file_path, build_index_from_scratch=True)

        # Create scan map ensuring pairs are iterable
        scan_map = {}
        for pair in scan_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                scan_map[pair[0]] = pair[1]
            else:
                print(f"Invalid scan pair format: {pair}")
                continue

        for spectrum in run:
            scan_id = int(spectrum['id'])
            if spectrum['ms level'] != 2:
                continue
            if scan_id in scan_map:
                oms_spec = oms.MSSpectrum()

                # Add metadata
                oms_spec.setNativeID(f"controllerType=0 controllerNumber=1 scan={scan_map[scan_id]}")
                oms_spec.setMSLevel(2)

                # Set retention time
                if spectrum.scan_time:
                    rt, unit = spectrum.scan_time
                    oms_spec.setRT(rt * 60 if unit == 'minute' else rt)

                # Add precursor info
                if spectrum.selected_precursors:
                    precursor = oms.Precursor()
                    precursor.setMZ(spectrum.selected_precursors[0]['mz'])
                    precursor.setCharge(spectrum.selected_precursors[0].get('charge', 1))
                    oms_spec.setPrecursors([precursor])

                # Add peaks
                raw_peaks = spectrum.peaks("centroided")

                if raw_peaks is not None and len(raw_peaks) > 0:
                    clean_peaks = []
                    for peak in raw_peaks:
                        if len(peak) != 2:
                            print(f"Skipping invalid peak tuple in {file_path} scan {scan_id}: {peak}")
                            continue
                        try:
                            mz_val, int_val = peak
                            # Ensure we have scalar values
                            if hasattr(mz_val, '__iter__') and not isinstance(mz_val, str):
                                mz_val = mz_val[0] if len(mz_val) > 0 else 0.0
                            if hasattr(int_val, '__iter__') and not isinstance(int_val, str):
                                int_val = int_val[0] if len(int_val) > 0 else 0.0

                            mz_f = float(mz_val)
                            int_f = float(int_val)
                            if np.isfinite(mz_f) and np.isfinite(int_f):
                                clean_peaks.append((mz_f, int_f))
                        except Exception as e:
                            print(f"Invalid peak values in {file_path} scan {scan_id}: {peak}, error: {e}")

                    if len(clean_peaks) > 0:
                        # Convert to numpy array and validate shape
                        peaks_array = np.array(clean_peaks, dtype=np.float64)
                        if peaks_array.ndim != 2 or peaks_array.shape[1] != 2:
                            print(f"Invalid peak array shape in {file_path} scan {scan_id}")
                            continue

                        mz = peaks_array[:, 0]
                        intensity = peaks_array[:, 1]

                        # Final validation check
                        if len(mz) != len(intensity):
                            print(f"Critical error: Array length mismatch in {file_path} scan {scan_id}")
                            continue

                        oms_spec.set_peaks((mz, intensity))
                        exp.addSpectrum(oms_spec)

        if exp.size() > 0:
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            output_path = temp_dir / f"{Path(file_path).stem}_{file_hash}_temp.mzML"
            oms.MzMLFile().store(str(output_path), exp)
            return str(output_path)

        if exp.size() == 0:
            print(f"[Info] No matching spectra found in file: {file_path}, skipping writing temp mzML.")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    return None


def _add_spectrum(experiment, sp_data, scan_id):
    """
    Helper to add a single MSSpectrum to an MSExperiment.
    """
    s = oms.MSSpectrum()
    s.setRT(float(sp_data.get('rtinseconds', 0)))
    s.setMSLevel(2)
    prec = oms.Precursor()
    mz = float(sp_data.get('pepmass', sp_data.get('PEPMASS', sp_data.get('precursor_mz', 0))))
    prec.setMZ(mz)
    ch_str = str(sp_data.get('charge', sp_data.get('CHARGE', '0'))).replace('+', '')
    if ch_str.isdigit():
        prec.setCharge(int(ch_str))
    else:
        prec.setCharge(0)
    s.setPrecursors([prec])
    # Use a more standard nativeID format for better compatibility
    s.setNativeID(f"controllerType=0 controllerNumber=1 scan={scan_id}")
    if 'peaks' in sp_data:
        mz_array, int_array = zip(*sp_data['peaks'])
        s.set_peaks([mz_array, int_array])
    experiment.addSpectrum(s)


def read_mgf(filename):
    """
    Minimal MGF reader for the representative falcon .mgf
    """
    spectra = []
    with open(filename, 'r') as f:
        spec = None
        for line in f:
            line = line.strip()
            if line == 'BEGIN IONS':
                spec = {'peaks': [], 'm/z array': [], 'intensity array': []}
            elif line == 'END IONS':
                spectra.append(spec)
                spec = None
            elif spec is not None:
                if '=' in line:
                    key, val = line.split('=', 1)
                    spec[key.lower()] = val
                else:
                    # Possibly a peak
                    parts = line.split()
                    if len(parts) == 2:
                        mz, intensity = parts
                        try:
                            spec['peaks'].append((float(mz), float(intensity)))
                            spec['m/z array'].append(float(mz))
                            spec['intensity array'].append(float(intensity))
                        except:
                            pass
    return spectra


##############################################################################
# 3) UPDATING + MERGING CLUSTER DICS
##############################################################################

def initial_cluster_dic(cluster_info_tsv, falcon_mgf, original_file_path):
    cluster_dic = {}
    singletons = []

    non_neg_rows = []
    neg_rows = []
    with open(cluster_info_tsv, 'r') as csvfile:
        rdr = csv.DictReader(csvfile, delimiter='\t')
        for row in rdr:
            cid = int(row['cluster'])
            if cid != -1:
                non_neg_rows.append(row)
            else:
                neg_rows.append(row)

    # Process non-neg clusters
    current_max_id = 0
    for row in non_neg_rows:
        cid = int(row['cluster'])
        current_max_id = max(current_max_id, cid)
        fn = row['filename']
        sc = int(row['scan'])
        fp = get_original_file_path(fn, original_file_path)
        if cid not in cluster_dic:
            cluster_dic[cid] = {'scan_list': []}
        cluster_dic[cid]['scan_list'].append((fp, sc))

    # Process neg clusters (singletons)
    for row in neg_rows:
        fn = row['filename']
        sc = int(row['scan'])
        fp = get_original_file_path(fn, original_file_path)
        singletons.append((fp, sc))

    # Attach Falcon MGF reps
    mgf_spectra = read_mgf(falcon_mgf)
    for spectrum in mgf_spectra:
        original_cid = int(spectrum['cluster'])
        if original_cid in cluster_dic:
            cluster_dic[original_cid]['spectrum'] = spectrum

    return cluster_dic, singletons


def update_cluster_dic(cluster_dic, cluster_info_tsv, falcon_mgf, original_file_path, path_map=None):
    """
    Merge new clustering results into existing cluster_dic, returning:
    - Updated cluster_dic (only non-singletons)
    - List of new singletons (filepath, scan) tuples
    """
    currentID_uniID = {}  # Mapping from Falcon's temporary IDs to unified cluster IDs
    new_singletons = []  # Stores (filepath, scan) of new singletons
    new_cluster_id = max(cluster_dic.keys(), default=0)

    # Categorize rows from cluster_info.tsv
    consensus_rows = []
    nonconsensus_rows = []

    with open(cluster_info_tsv, 'r') as csvfile:
        rdr = csv.DictReader(csvfile, delimiter='\t')
        for row in rdr:
            cid = int(row['cluster'])
            if row['filename'] == "consensus.mzML":
                consensus_rows.append(row)
            elif cid != -1:
                nonconsensus_rows.append(row)
            else:
                # Collect singletons directly
                # Resolve singleton paths using path_map
                if '_' in row['scan'] and path_map is not None:  # Format: HASH_basename_scan
                    path_part = '_'.join(row['scan'].split('_')[:-1])
                    try:
                        fp = path_map.loc[path_part, "full_path"]
                        sc = int(row['scan'].split('_')[-1])
                        new_singletons.append((fp, sc))
                    except KeyError:
                        print(f"Path mapping missing for {path_part}")
                else:  # New singletons from current batch
                    fp = get_original_file_path(row['filename'], original_file_path)
                    new_singletons.append((fp, int(row['scan'])))

    # Process consensus spectra (from previous batches)
    for row in consensus_rows:
        cid = int(row['cluster'])
        scan_str = row['scan']

        # Robustly parse the scan ID, which may have format 'key=value' or 'value_sample'
        try:
            # Handle cases like 'scan=123' or 'trum=8181043'
            if '=' in scan_str:
                scan_str = scan_str.split('=')[-1]

            # Handle cases like '123_1' by taking the first part
            scan_parts = scan_str.split('_')
            sc = int(scan_parts[0])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse scan ID '{row['scan']}' for consensus spectrum. Skipping row.")
            continue  # Skip this problematic row

        if cid not in currentID_uniID:
            currentID_uniID[cid] = sc

    # Process non-consensus non-singleton clusters
    for row in tqdm(nonconsensus_rows, desc="Merging non-singletons"):
        cid = int(row['cluster'])
        fn = row['filename']
        scan_str = row['scan']
        if '_' in scan_str and path_map is not None:
            try:
                # Split composite ID: "b5d82eac_H11-2_3083" -> ("b5d82eac_H11-2", 3083)
                path_part, sc = scan_str.rsplit('_', 1)
                sc = int(sc)
                fp = path_map.loc[path_part, "full_path"]
            except (ValueError, KeyError) as e:
                print(f"Invalid composite scan {scan_str}: {e}")
                continue
        else:
            # Handle normal integer scans from raw files
            try:
                sc = int(scan_str)
                fp = get_original_file_path(fn, original_file_path)
            except ValueError:
                print(f"Invalid scan value {scan_str} in file {fn}")
                continue

        # Get or create unified cluster ID
        if cid not in currentID_uniID:
            new_cluster_id += 1
            currentID_uniID[cid] = new_cluster_id
            cluster_dic[new_cluster_id] = {'scan_list': [], 'spec_pool': []}
        mapped_cid = currentID_uniID[cid]
        cluster_dic[mapped_cid]['scan_list'].append((fp, sc))

    # Attach Falcon MGF representatives (skip singletons)
    mgf_spectra = read_mgf(falcon_mgf)
    for spectrum in mgf_spectra:
        try:
            old_cid = int(spectrum.get('cluster', -1))
            if old_cid == -1:
                continue  # Skip singleton representatives

            if old_cid in currentID_uniID:
                unified_cid = currentID_uniID[old_cid]
                cluster_dic[unified_cid]['spectrum'] = spectrum
        except (ValueError, KeyError):
            continue

    return cluster_dic, new_singletons


def save_cluster_dic_optimized(cluster_dic, out_dir):
    """Optimized saving with binary peak storage and Arrow-native parallelism"""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save ALL scan lists regardless of spectrum presence
    scan_data = []
    for cid, cdata in cluster_dic.items():
        for fp, sc in cdata.get("scan_list", []):
            scan_data.append({
                "cluster_id": cid,
                "filename": fp,
                "scan": sc
            })

    scan_table = pa.Table.from_pylist(scan_data)
    feather.write_feather(scan_table, os.path.join(out_dir, "scan_list.feather"))

    # 2. Save spectrum data only when present
    spec_data = []
    for cid, cdata in cluster_dic.items():
        spectrum = cdata.get("spectrum")
        if spectrum and 'peaks' in spectrum and len(spectrum['peaks']) > 0:
            peaks = np.array(spectrum['peaks'], dtype=np.float32)
            ch_str = str(spectrum.get('charge', '0')).replace('+', '')
            charge_val = int(ch_str) if ch_str.isdigit() else 0
            spec_data.append({
                "cluster_id": cid,
                "spectrum_mz": peaks[:, 0].tobytes(),
                "spectrum_intensity": peaks[:, 1].tobytes(),
                "precursor_mz": np.float32(spectrum.get('precursor_mz', 0)),
                "rtinseconds": np.float32(spectrum.get('rtinseconds', 0)),
                "charge": np.int32(charge_val),
                "title": str(spectrum.get('title', ''))
            })

    if spec_data:
        schema = pa.schema([
            ('cluster_id', pa.int32()),
            ('spectrum_mz', pa.binary()),
            ('spectrum_intensity', pa.binary()),
            ('precursor_mz', pa.float32()),
            ('rtinseconds', pa.float32()),
            ('charge', pa.int32()),
            ('title', pa.string())
        ])
        spec_table = pa.Table.from_pylist(spec_data, schema=schema)
        pq.write_table(
            spec_table,
            os.path.join(out_dir, "spec_data.parquet"),
            compression='zstd',
            compression_level=3
        )


def load_cluster_dic_optimized(in_dir):
    """Optimized loading with binary deserialization"""
    cluster_dic = {}
    if not os.path.exists(in_dir):
        return cluster_dic

    # 1. Load ALL scan lists first
    scan_path = os.path.join(in_dir, "scan_list.feather")
    if os.path.exists(scan_path):
        scan_df = feather.read_table(scan_path).to_pandas()
        for row in scan_df.itertuples():
            cluster_dic.setdefault(row.cluster_id, {'scan_list': []})
            cluster_dic[row.cluster_id]['scan_list'].append((row.filename, row.scan))

    # 2. Add spectra to existing clusters where available
    cluster_path = os.path.join(in_dir, "spec_data.parquet")
    if os.path.exists(cluster_path):
        cluster_df = pq.read_table(cluster_path).to_pandas()
        for row in cluster_df.itertuples():
            if row.cluster_id in cluster_dic:
                mz = np.frombuffer(row.spectrum_mz, dtype=np.float32)
                intensity = np.frombuffer(row.spectrum_intensity, dtype=np.float32)
                cluster_dic[row.cluster_id]['spectrum'] = {
                    'peaks': list(zip(mz, intensity)),
                    'precursor_mz': row.precursor_mz,
                    'rtinseconds': row.rtinseconds,
                    'charge': row.charge,
                    'title': row.title
                }

    return cluster_dic


def _custom_merge_mzml_files(file_list, output_path, threads=8):
    """
    Legacy custom mzML merger (kept for compatibility).
    Use _custom_merge_mzml_files_parallel for better performance.
    """
    return _custom_merge_mzml_files_parallel(file_list, output_path, threads)


##############################################################################
# 4) DRIVER LOGIC: ONE FOLDER AT A TIME
##############################################################################

def cluster_one_folder(folder, checkpoint_dir, output_dir, tool_dir, precursor_tol, fragment_tol, min_mz_range, min_mz,
                       max_mz, eps):
    """
    Given a single folder that has *.mzML files,
    - if 'cluster_dic.h5' doesn't exist => run initial clustering
    - else => run incremental clustering
    Then update cluster_dic.h5 and consensus.mzML in checkpoint_dir.
    """

    # time recording structure
    start_time = time.time()
    start_cpu_time = time.process_time()
    timing_log = {}

    consensus_path = os.path.join(checkpoint_dir, "consensus.mzML")

    output_consensus_path = os.path.join(output_dir, "consensus.mzML")

    scan_feather = os.path.join(checkpoint_dir, "scan_list.feather")
    spec_parquet = os.path.join(checkpoint_dir, "spec_data.parquet")

    has_checkpoint = os.path.exists(scan_feather) and os.path.exists(spec_parquet)

    if not has_checkpoint:
        # run falcon
        print("[Initial] Running Falcon on all new spectra...")
        input_files = f"{folder}/*.mzML"
        run_falcon(input_files, "falcon", precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps)
        falcon_end_time = time.time()
        timing_log['Falcon initial'] = falcon_end_time - start_time
        print(f"Falcon clustering took {falcon_end_time - start_time:.2f} s.")

        # summarize results
        cluster_info_tsv = summarize_output(output_dir, os.path.join(tool_dir, 'summarize_results.py'), 'falcon.csv')
        summarize_time = time.time()
        timing_log['summarize initial results'] = summarize_time - falcon_end_time
        print(f"Summarize falcon results took {summarize_time - falcon_end_time:.2f} s.")

        # initialize cluster dic
        falcon_mgf_path = os.path.join(os.getcwd(), "falcon.mgf")
        print("[cluster_one_folder] Performing initial clustering...")
        cluster_dic, singletons = initial_cluster_dic(cluster_info_tsv, falcon_mgf_path, folder)
        cluster_dic_time = time.time()
        timing_log['Initial cluster dic'] = cluster_dic_time - summarize_time
        print(f"Cluster dic initialize took {cluster_dic_time - summarize_time:.2f} s.")

        # write consensus mzML file
        n_written = write_mzml(cluster_dic, output_consensus_path)
        consensus_end_time = time.time()
        timing_log['Initial round consensus write'] = consensus_end_time - cluster_dic_time
        print(f"Write consensus mzML file took {consensus_end_time - cluster_dic_time:.2f} s.")

        # write singletons
        singletons_mzml_path = os.path.join(output_dir, "singletons.mzML")
        write_singletons_mzml(singletons, singletons_mzml_path)
    else:
        # Incremental processing
        # First Falcon run: new data + consensus
        print("[Incremental] Phase 1: non-singleton clustering...")
        input_files = f"{folder}/*.mzML {consensus_path}"
        run_falcon(input_files, "falcon1", precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps)
        falcon1_end_time = time.time()
        timing_log['Falcon phase 1 clustering'] = falcon1_end_time - start_time
        print(f"Falcon phase 1 clustering took {falcon1_end_time - start_time:.2f} s.")

        # summarize phase1 results
        cluster_info1_tsv = summarize_output(output_dir, os.path.join(tool_dir, 'summarize_results.py'),
                                             falcon_csv="falcon1.csv")
        summarize1_time = time.time()
        timing_log['Summarize falcon phase 1 results'] = summarize1_time - falcon1_end_time
        print(f"Summarize falcon phase 1 results took {summarize1_time - falcon1_end_time:.2f} s.")

        # load cluster dic time
        cluster_dic = load_cluster_dic_optimized(checkpoint_dir)
        load_cluster_dic_time = time.time()
        timing_log['Load cluster dic'] = load_cluster_dic_time - summarize1_time
        print(f"Load cluster dic took {load_cluster_dic_time - summarize1_time:.2f} s.")

        # Update cluster dic phase 1
        cluster_dic, new_singletons1 = update_cluster_dic(cluster_dic, cluster_info1_tsv, "falcon1.mgf", folder)
        update1_cluster_dic_time = time.time()
        timing_log['Update cluster dic phase 1'] = update1_cluster_dic_time - load_cluster_dic_time
        print(f"Update cluster dic phase1 took {update1_cluster_dic_time - load_cluster_dic_time:.2f} s.")

        # write phase1 singletons
        temp_singletons_path = os.path.join(output_dir, "temp_singletons.mzML")
        write_singletons_mzml(new_singletons1, temp_singletons_path)
        write_singletons1_end_time = time.time()
        timing_log['Write phase 1 singletons'] = write_singletons1_end_time - update1_cluster_dic_time
        print(f"Write phase 1 singletons took {write_singletons1_end_time - update1_cluster_dic_time:.2f} s.")

        # Second Falcon run: existing singletons + new temp_singletons
        singletons_mzml_path = os.path.join(checkpoint_dir, "singletons.mzML")
        run_falcon(f"{singletons_mzml_path} {temp_singletons_path}", "falcon2", precursor_tol, fragment_tol,
                   min_mz_range, min_mz, max_mz, eps)
        falcon2_end_time = time.time()
        timing_log['Falcon phase 2 clustering'] = falcon2_end_time - write_singletons1_end_time
        print(f"Falcon phase 2 took {falcon2_end_time - write_singletons1_end_time:.2f} s.")

        # summarize phase2 results
        cluster_info2_tsv = summarize_output(output_dir, os.path.join(tool_dir, 'summarize_results.py'),
                                             falcon_csv="falcon2.csv")
        summarize2_time = time.time()
        timing_log['Summarize falcon phase 2 results'] = summarize2_time - falcon2_end_time
        print(f"Summarize falcon phase 2 results took {summarize2_time - falcon2_end_time:.2f} s.")

        # update cluster dic phase 2
        # Load both path maps
        existing_map_path = Path(singletons_mzml_path).with_suffix(".pathmap.feather")
        temp_map_path = Path(temp_singletons_path).with_suffix(".pathmap.feather")
        # Handle missing maps gracefully
        path_maps = []
        if existing_map_path.exists():
            path_maps.append(feather.read_feather(existing_map_path))
        if temp_map_path.exists():
            path_maps.append(feather.read_feather(temp_map_path))
        # Combine maps if both exist
        combined_map = pd.concat(path_maps).drop_duplicates() if path_maps else None
        cluster_dic, new_singletons2 = update_cluster_dic(cluster_dic, cluster_info2_tsv, "falcon2.mgf", folder,
                                                          combined_map)
        update2_cluster_dic_time = time.time()
        timing_log['Update cluster dic phase2'] = update2_cluster_dic_time - summarize2_time
        print(f"Update cluster dic phase2 took {update2_cluster_dic_time - summarize2_time:.2f} s.")

        # wirte final consensus results
        n_written = write_mzml(cluster_dic, output_consensus_path)
        consensus_end_time = time.time()
        timing_log['Write consensus mzML file'] = consensus_end_time - update2_cluster_dic_time
        print(f"Write consensus mzML file took {consensus_end_time - update2_cluster_dic_time:.2f} s.")

        # write phase2 singletons
        singletons2_mzml_path = os.path.join(output_dir, "singletons.mzML")
        write_singletons_mzml(new_singletons2, singletons2_mzml_path)
        write_singletons2_end_time = time.time()
        timing_log['Write phase2 singletons'] = write_singletons2_end_time - consensus_end_time
        print(f"Write phase2 singletons took {write_singletons2_end_time - consensus_end_time:.2f} s.")

    # Clean up local falcon outputs
    # if os.path.exists("falcon.csv"):
    #     os.remove("falcon.csv")
    # if os.path.exists("falcon.mgf"):
    #     os.remove("falcon.mgf")
    # if os.path.exists("output_summary"):
    #     shutil.rmtree("output_summary", ignore_errors=True)

    # save updated cluster_dic
    cluster_end_time = time.time()
    save_cluster_dic_optimized(cluster_dic, output_dir)
    print(f"[cluster_one_folder] Updated cluster_dic saved at {output_dir}")
    save_cluster_dic_end_time = time.time()
    timing_log['Save cluster dic'] = save_cluster_dic_end_time - cluster_end_time
    print(f"Save cluster dic took {save_cluster_dic_end_time - cluster_end_time:.2f} s.")

    # final results output
    current_batch_files = set()
    for fname in os.listdir(folder):
        if fname.endswith('.mzML') and fname != 'consensus.mzML':
            current_batch_files.add(fname)  # Store base name, not full path
    finalize_results(cluster_dic, output_dir, current_batch_files)
    output_results_end_time = time.time()
    timing_log['Output results'] = output_results_end_time - save_cluster_dic_end_time
    print(f"Output results took {output_results_end_time - save_cluster_dic_end_time:.2f} s.")

    end_time = time.time()
    end_cpu_time = time.process_time()

    timing_log["Total wall time (hours)"] = (end_time - start_time) / 3600
    timing_log["Total CPU time (hours)"] = (end_cpu_time - start_cpu_time) / 3600

    # Write to file
    timing_file = os.path.join(output_dir, "timing_report.txt")
    with open(timing_file, "w") as f:
        f.write(f"Run Timestamp: {datetime.datetime.now()}\n")
        for k, v in timing_log.items():
            f.write(f"{k}: {v:.2f}\n")

    print(f"[Timing] Full timing report written to: {timing_file}")


def finalize_results(cluster_dic, output_dir, current_batch_files=None):
    out_tsv = os.path.join(output_dir, "cluster_info.tsv")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ['cluster', 'filename', 'scan', 'new_batch']
    with open(out_tsv, 'w', newline='') as f:
        w = DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for cid, cdata in cluster_dic.items():
            for item in cdata['scan_list']:
                fn, sc = item
                is_new = 'yes' if current_batch_files and os.path.basename(fn) in current_batch_files else 'no'
                row = {
                    'cluster': cid,
                    'filename': os.path.basename(fn),
                    'scan': sc,
                    'new_batch': is_new
                }
                w.writerow(row)

    print(f"[finalize_results] Wrote final cluster_info.tsv: {out_tsv}")


##############################################################################
# 5) MAIN
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to a folder containing *.mzML")
    parser.add_argument("--checkpoint_dir", default="./checkpoint",
                        help="Where cluster_dic.h5 and consensus.mzML are stored/updated.")
    parser.add_argument("--output_dir", default="./results",
                        help="Where a final cluster_info.tsv might be placed (after all runs).")
    parser.add_argument("--tool_dir", default="./bin",
                        help="Where tool scripts might be placed.")

    # Falcon parameters
    parser.add_argument("--precursor_tol", default="20 ppm",
                        help="Precursor tolerance (e.g., '20 ppm' or '0.5 Da')")
    parser.add_argument("--fragment_tol", type=float, default=0.05,
                        help="Fragment tolerance in Da")
    parser.add_argument("--min_mz_range", type=float, default=0,
                        help="Minimum m/z range for clustering")
    parser.add_argument("--min_mz", type=float, default=0,
                        help="Minimum m/z value to consider")
    parser.add_argument("--max_mz", type=float, default=4000,
                        help="Maximum m/z value to consider")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="EPS parameter for DBSCAN clustering")
    args = parser.parse_args()

    # print the version of the script
    # define the version of the script by "date_version"
    __version__ = f"{datetime.datetime.now().strftime('%Y%m%d')}_1.1.0"
    print(f"Running version: {__version__}")

    # 1) Process exactly one folder
    cluster_one_folder(
        args.folder,
        args.checkpoint_dir,
        args.output_dir,
        args.tool_dir,
        args.precursor_tol,
        args.fragment_tol,
        args.min_mz_range,
        args.min_mz,
        args.max_mz,
        args.eps
    )


if __name__ == "__main__":
    main()
