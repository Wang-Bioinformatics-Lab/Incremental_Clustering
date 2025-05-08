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
        f"--eps {eps}"
    )
    print(f"[run_falcon] Running: {command}")
    process = subprocess.Popen(command, shell=True)
    retcode = process.wait()
    # if retcode != 0:
    #     raise RuntimeError(f"Falcon failed with exit code {retcode}")


def summarize_output(output_path,summarize_script="summarize_results.py", falcon_csv="falcon.csv"):
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

def get_original_file_path(filename,original_filepah):
    return os.path.join(original_filepah,filename)

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

def write_mzml(cluster_dic, out_path):

    ref_write = {} #cid:spectrum data
    fetch_write = defaultdict(list) #filepath:(scan in original file, scan id need to be written into the consensus.mzML file)
    for cid, data in cluster_dic.items():
        n_pool = len(data['scan_list'])
        if (n_pool <= 5) and (n_pool > 1):
            if 'spectrum' in data:
                ref_write[cid] = data['spectrum']
        if (n_pool == 1):
            if data['scan_list']:  # Check if scan list is not empty
                fp, sc = data['scan_list'][0]  # Unpack first (only) entry
                fetch_write[fp].append((sc, cid))
        if (n_pool > 5):
            chosen = random.sample(data['scan_list'], 5)
            if 'spectrum' in data:
                ref_write[cid] = data['spectrum']
            _id = 1
            for c in chosen:
                new_id = f"{cid}_{_id}"
                fetch_write[c[0]].append((c[1],new_id))
                _id = _id + 1

    temp_dir = Path(out_path).parent / "temp_consensus"
    temp_dir.mkdir(exist_ok=True)

    # Process reference spectra
    exp_final = oms.MSExperiment()
    if ref_write:
        exp_ref = oms.MSExperiment()
        for cid, sp_data in ref_write.items():
            _add_spectrum(exp_ref, sp_data, cid)
        exp_final = exp_ref  # Initialize with reference spectra

    # Process fetched spectra
    if fetch_write:
        tasks = []
        for fp, scans in fetch_write.items():
            if Path(fp).exists():
                tasks.append((fp, scans, temp_dir))
            else:
                print(f"File not found: {fp}")

        with Pool(processes=min(16, os.cpu_count())) as pool:
            temp_files = pool.starmap(_process_fetch_file, tasks)

        # Merge results using OpenMS tools
        for temp_file in filter(None, temp_files):
            temp_path = Path(temp_file)
            if temp_path.exists():
                temp_exp = oms.MSExperiment()
                oms.MzMLFile().load(str(temp_path), temp_exp)
                for spec in temp_exp.getSpectra():
                    exp_final.addSpectrum(spec)
                temp_path.unlink()
            else:
                print(f"[Warning] Temp file not found (skipping): {temp_file}")

    # Write final output
    oms.MzMLFile().store(out_path, exp_final)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return exp_final.size()


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
                oms_spec.setNativeID(f"scan={scan_map[scan_id]}")
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
                            mz_f = float(mz_val)
                            int_f = float(int_val)
                            if np.isfinite(mz_f) and np.isfinite(int_f):
                                clean_peaks.append((mz_f, int_f))
                        except:
                            print(f"Invalid peak values in {file_path} scan {scan_id}: {peak}")

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
    ch_str = str(sp_data.get('charge', sp_data.get('CHARGE', '0'))).replace('+','')
    if ch_str.isdigit():
        prec.setCharge(int(ch_str))
    else:
        prec.setCharge(0)
    s.setPrecursors([prec])
    s.setNativeID(f"scan={scan_id}")
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
                    key, val = line.split('=',1)
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
    """
    Create a cluster_dic from scratch, handling cluster IDs:
    - Non-`-1` clusters retain their original IDs.
    - `-1` clusters get new sequential IDs.
    """
    cluster_dic = {}
    current_max_id = 0  # Track the highest cluster ID

    # Split rows into non-neg and neg clusters
    non_neg_rows = []
    neg_rows = []
    with open(cluster_info_tsv, 'r') as csvfile:
        rdr = csv.DictReader(csvfile, delimiter='\t')
        for row in rdr:
            cid = int(row['cluster'])
            if cid != -1:
                non_neg_rows.append(row)
                current_max_id = max(current_max_id, cid)  # Track max ID
            else:
                neg_rows.append(row)

    # Process non-neg clusters (original IDs)
    for row in non_neg_rows:
        cid = int(row['cluster'])
        fn = row['filename']
        sc = int(row['scan'])
        # pmz = row['precursor_mz']
        # rt = row['retention_time']
        fp = get_original_file_path(fn,original_file_path)


        if cid not in cluster_dic:
            cluster_dic[cid] = {'scan_list': [], 'spec_pool': []}
        cluster_dic[cid]['scan_list'].append((fp, sc))

    # Process neg clusters (assign new IDs sequentially)
    for row in neg_rows:
        current_max_id += 1
        new_cid = current_max_id
        fn = row['filename']
        sc = int(row['scan'])
        # pmz = row['precursor_mz']
        # rt = row['retention_time']
        fp = get_original_file_path(fn, original_file_path)

        cluster_dic[new_cid] = {
            'scan_list': [(fp, sc)],
            'spec_pool': [],
        }

    # Attach Falcon MGF reps (use original IDs directly)
    mgf_spectra = read_mgf(falcon_mgf)
    for spectrum in mgf_spectra:
        original_cid = int(spectrum['cluster'])
        if original_cid in cluster_dic:
            cluster_dic[original_cid]['spectrum'] = spectrum
        else:
            print(f"Cluster {original_cid} not found, skipping.")

    return cluster_dic

def update_cluster_dic(cluster_dic, cluster_info_tsv, falcon_mgf, original_file_path):
    """
    Merge newly generated cluster_info into existing cluster_dic
    (i.e., "incremental" approach).
    """
    currentID_uniID = {}
    cat_consensus = []
    cat_noncons_nonneg = []
    cat_neg = []

    with open(cluster_info_tsv, 'r') as csvfile:
        rdr = csv.DictReader(csvfile, delimiter='\t')
        for row in rdr:
            cid = int(row['cluster'])
            fn  = row['filename']
            if fn == "consensus.mzML":
                cat_consensus.append(row)
            elif cid != -1:
                cat_noncons_nonneg.append(row)
            else:
                cat_neg.append(row)

    # 1) consensus
    for row in cat_consensus:
        cid = int(row['cluster'])
        sc  = row['scan'].split('_')[0]
        if sc.isdigit():
            sc = int(sc)
        if cid not in currentID_uniID:
            currentID_uniID[cid] = sc

    # 2) non-consensus, non-neg
    new_key = max(cluster_dic.keys(), default=0)
    for row in tqdm(cat_noncons_nonneg):
        cid = int(row['cluster'])
        fn = row['filename']
        sc = int(row['scan'])
        # pmz= row['precursor_mz']
        # rt = row['retention_time']
        fp = get_original_file_path(fn, original_file_path)

        # try:
        #     sp_data = indexer.get_spectrum(base, sc)
        # except Exception as e:
        #     print("error fetching spectrum", base, sc)

        if cid not in currentID_uniID:
            new_key += 1
            currentID_uniID[cid] = new_key
            cluster_dic[new_key] = {'scan_list':[], 'spec_pool':[]}
        mapped_cid = currentID_uniID[cid]
        cluster_dic[mapped_cid]['scan_list'].append((fp, sc))


    # 3) cluster_id = -1
    new_cluster_id = max(cluster_dic.keys(), default=0)
    for row in tqdm(cat_neg):
        fn = row['filename']
        sc = int(row['scan'])
        # pmz= row['precursor_mz']
        # rt = row['retention_time']
        fp = get_original_file_path(fn, original_file_path)

        new_cluster_id += 1
        currentID_uniID[new_cluster_id] = new_cluster_id
        cluster_dic[new_cluster_id] = {'scan_list':[], 'spec_pool':[]}
        cluster_dic[new_cluster_id]['scan_list'].append((fp, sc))

    # attach falcon mgf reps
    mgf_spectra = read_mgf(falcon_mgf)
    for s in mgf_spectra:
        try:
            old_cid = int(s['cluster'])
            if old_cid in currentID_uniID:
                new_cid = currentID_uniID[old_cid]
                cluster_dic[new_cid]['spectrum'] = s
        except KeyError:
            print(old_cid, s['title'])

    return cluster_dic


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
##############################################################################
# 4) DRIVER LOGIC: ONE FOLDER AT A TIME
##############################################################################

def cluster_one_folder(folder, checkpoint_dir, output_dir, tool_dir, precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps):
    """
    Given a single folder that has *.mzML files,
    - if 'cluster_dic.h5' doesn't exist => run initial clustering
    - else => run incremental clustering
    Then update cluster_dic.h5 and consensus.mzML in checkpoint_dir.
    """

    consensus_path   = os.path.join(checkpoint_dir, "consensus.mzML")

    output_consensus_path = os.path.join(output_dir, "consensus.mzML")

    scan_feather = os.path.join(checkpoint_dir, "scan_list.feather")
    spec_parquet = os.path.join(checkpoint_dir, "spec_data.parquet")



    if os.path.exists(consensus_path):
        # Include the consensus file along with the new mzML files.
        input_files = f"{folder}/*.mzML {consensus_path}"
        print(f"[cluster_one_folder] Consensus file found; adding {consensus_path} to Falcon input.")
    else:
        input_files = f"{folder}/*.mzML"
        print(f"[cluster_one_folder] No consensus file found; proceeding with folder mzML files only.")

    current_batch_files = set()
    for fname in os.listdir(folder):
        if fname.endswith('.mzML') and fname != 'consensus.mzML':
            current_batch_files.add(fname)  # Store base name, not full path

    start_time = time.time()

    # 1) RUN FALCON on the folder's *.mzML
    run_falcon(
        input_files,
        "falcon",
        precursor_tol=precursor_tol,
        fragment_tol=fragment_tol,
        min_mz_range=min_mz_range,
        min_mz=min_mz,
        max_mz=max_mz,
        eps=eps
    )

    falcon_end_time = time.time()

    print(f"Falcon clustering took {falcon_end_time- start_time:.2f} s.")

    # Build a dictionary of the spectra in this folder
    # print(f"[cluster_one_folder] Building mzML index for folder: {folder}")
    # indexer = create_mzml_indexer(folder)
    print(f"[cluster_one_folder] Building spectra dict for folder: {folder}")
    # folder_spec_dic = read_mzml_parallel(folder)

    mzml_end_time = time.time()

    print(f"Reading files took {mzml_end_time - falcon_end_time:.2f} s.")

    # Read existing cluster_dic (if any)

    has_checkpoint = os.path.exists(scan_feather) and os.path.exists(spec_parquet)

    if has_checkpoint:
        cluster_dic = load_cluster_dic_optimized(checkpoint_dir)
    else:
        cluster_dic = {}  # Start fresh

    cluster_dic_load_time = time.time()

    print(f"Loading clustering dic took {cluster_dic_load_time - mzml_end_time:.2f} s.")

    # 2) Summarize => we get cluster_info.tsv
    cluster_info_tsv = summarize_output(output_dir,summarize_script= os.path.join(tool_dir,"summarize_results.py"), falcon_csv="falcon.csv")

    sumarize_results_end_time = time.time()
    print(f"sumarize results took {sumarize_results_end_time - cluster_dic_load_time:.2f} s.")

    # 3) Merge or Initialize cluster_dic
    falcon_mgf_path = os.path.join(os.getcwd(), "falcon.mgf")

    if has_checkpoint:
        # incremental
        print("[cluster_one_folder] Performing incremental clustering...")
        cluster_dic = update_cluster_dic(cluster_dic, cluster_info_tsv, falcon_mgf_path, folder)
    else:
        # initial
        print("[cluster_one_folder] Performing initial clustering...")
        cluster_dic = initial_cluster_dic(cluster_info_tsv, falcon_mgf_path, folder)

    update_cluster_end_time = time.time()
    print(f"Merge results took {update_cluster_end_time - sumarize_results_end_time:.2f} s.")
    # del folder_spec_dic  # Explicitly delete the dictionary
    # gc.collect()  # Force garbage collection

    #indexer.close()


    # Clean up local falcon outputs
    # if os.path.exists("falcon.csv"):
    #     os.remove("falcon.csv")
    # if os.path.exists("falcon.mgf"):
    #     os.remove("falcon.mgf")
    # if os.path.exists("output_summary"):
    #     shutil.rmtree("output_summary", ignore_errors=True)

    # 4) Write updated consensus
    n_written = write_mzml(cluster_dic, output_consensus_path)
    print(f"[cluster_one_folder] Wrote {n_written} spectra to consensus: {output_consensus_path}")

    consensus_write_end_time = time.time()
    print(f"consensus mzML writing took {consensus_write_end_time - update_cluster_end_time:.2f} s.")


    # 5) Save updated cluster_dic
    save_cluster_dic_optimized(cluster_dic, output_dir)
    print(f"[cluster_one_folder] Updated cluster_dic saved at {output_dir}")

    save_cluster_dic_end_time = time.time()
    print(f"Save cluster dic took {save_cluster_dic_end_time - consensus_write_end_time:.2f} s.")


    finalize_results(cluster_dic, output_dir, current_batch_files)
    output_results_end_time = time.time()
    print(f"Output results took {output_results_end_time - save_cluster_dic_end_time:.2f} s.")

    # optional memory cleanup
    # del cluster_dic, folder_spec_dic
    # gc.collect()

# def finalize_results(cluster_dic, output_dir, current_batch_files=None):
#     out_tsv = os.path.join(output_dir, "cluster_info.tsv")
#     os.makedirs(output_dir, exist_ok=True)
#
#     fieldnames = ['precursor_mz','retention_time','cluster','filename','scan', 'new_batch']
#     with open(out_tsv, 'w', newline='') as f:
#         w = DictWriter(f, fieldnames=fieldnames, delimiter='\t')
#         w.writeheader()
#         for cid, cdata in cluster_dic.items():
#             for item in cdata['scan_list']:
#                 fn, sc, pmz, rt = item
#                 is_new = 'yes' if current_batch_files and os.path.basename(fn) in current_batch_files else 'no'
#                 row = {
#                     'precursor_mz': pmz,
#                     'retention_time': rt,
#                     'cluster': cid,
#                     'filename': fn,
#                     'scan': sc,
#                     'new_batch': is_new
#                 }
#                 w.writerow(row)
#
#     print(f"[finalize_results] Wrote final cluster_info.tsv: {out_tsv}")

def finalize_results(cluster_dic, output_dir, current_batch_files=None):
    out_tsv = os.path.join(output_dir, "cluster_info.tsv")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ['cluster','filename','scan', 'new_batch']
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
    parser.add_argument("--eps", type=float, default=0.1,
                       help="EPS parameter for DBSCAN clustering")
    args = parser.parse_args()

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
