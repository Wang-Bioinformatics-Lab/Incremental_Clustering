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
from numcodecs import VLenArray, Blosc  # Add these imports at the top
import zarr
import numcodecs
import pyarrow as pa
from zarr.storage import ZipStore
import concurrent.futures
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
    if retcode != 0:
        raise RuntimeError(f"Falcon failed with exit code {retcode}")


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
                'm/z array': [],
                'intensity array': [],
                'precursor_mz': spectrum.selected_precursors[0]['mz'],
                'rtinseconds': spectrum.scan_time[0],
                'scans': spectrum['id'],
                'charge': spectrum.selected_precursors[0].get('charge', None)
            }
            for mz, intensity in spectrum.peaks("centroided"):
                spectrum_dict['peaks'].append((mz, intensity))
                spectrum_dict['m/z array'].append(mz)
                spectrum_dict['intensity array'].append(intensity)
            spectra.append(spectrum_dict)
    return spectra

def read_mzml_parallel(folder_path, max_workers=4):
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
    """
    Writes a "consensus.mzML" from the cluster_dic. If a cluster has >=10 spectra,
    it samples them. Returns count of total spectra written.
    """
    exp = oms.MSExperiment()
    write_count = 0
    for cid, data in cluster_dic.items():
        try:
            n_pool = len(data['spec_pool'])
            if n_pool < 10:
                # Use single 'spectrum'
                sp_data = data['spectrum']
                _add_spectrum(exp, sp_data, cid)
                write_count += 1
            else:
                # sample 10 from spec_pool + the 'spectrum'
                chosen = random.sample(data['spec_pool'], 10)
                chosen.append(data['spectrum'])
                for i, sp_data in enumerate(chosen, start=1):
                    new_id = f"{cid}_{i}"
                    _add_spectrum(exp, sp_data, new_id)
                write_count += 11
                data['spec_pool'] = chosen
        except Exception as err:
            print(f"Error in write_mzml for cluster {cid}: {err}")
    f = oms.MzMLFile()
    f.store(out_path, exp)
    return write_count

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
# 2) CLUSTER_DICT Checkpointing
##############################################################################

def save_cluster_dic_hdf5_optimized(cluster_dic, out_path):
    with h5py.File(out_path, "w") as f:
        for cid, cdata in cluster_dic.items():
            grp = f.create_group(str(cid))
            # scan_list
            scan_list_data = []
            for item in cdata.get("scan_list", []):
                fn, sc, pmz, rt = item
                scan_list_data.append((fn, sc, float(pmz), float(rt)))
            if scan_list_data:
                dt = np.dtype([
                    ("filename","S256"),
                    ("scan","i4"),
                    ("precursor_mz","f8"),
                    ("retention_time","f8")
                ])
                arr = np.array(scan_list_data, dtype=dt)
                grp.create_dataset("scan_list", data=arr, compression="gzip", chunks=True)

            # spec_pool
            sp_pool_data = [
                np.frombuffer(pickle.dumps(x), dtype="uint8")
                for x in cdata.get("spec_pool", [])
            ]
            if sp_pool_data:
                ds = grp.create_dataset(
                    "spec_pool",
                    (len(sp_pool_data),),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                    compression="gzip"
                )
                ds[...] = sp_pool_data

            # representative 'spectrum'
            if "spectrum" in cdata:
                ser = pickle.dumps(cdata["spectrum"])
                grp.create_dataset("spectrum", data=np.frombuffer(ser, dtype="uint8"))

            if "title" in cdata:
                grp.attrs["title"] = cdata["title"]

def load_cluster_dic_hdf5_optimized(in_path):
    cluster_dic = {}
    if not os.path.exists(in_path):
        return cluster_dic

    with h5py.File(in_path, "r") as f:
        for cid in f:
            grp = f[cid]
            cid_int = int(cid)
            cluster_dic[cid_int] = {
                "scan_list": [],
                "spec_pool": [],
                "spectrum": {}
            }
            if "scan_list" in grp:
                arr = grp["scan_list"][...]
                for row in arr:
                    fn   = row["filename"].decode("utf-8")
                    sc   = row["scan"]
                    pmz  = row["precursor_mz"]
                    rt   = row["retention_time"]
                    cluster_dic[cid_int]["scan_list"].append((fn, sc, pmz, rt))

            if "spec_pool" in grp:
                for item in grp["spec_pool"]:
                    spd = pickle.loads(item.tobytes())
                    cluster_dic[cid_int]["spec_pool"].append(spd)

            if "spectrum" in grp:
                sp = grp["spectrum"][()]
                cluster_dic[cid_int]["spectrum"] = pickle.loads(sp.tobytes())

            if "title" in grp.attrs:
                cluster_dic[cid_int]["title"] = grp.attrs["title"]

    return cluster_dic

##############################################################################
# 3) UPDATING + MERGING CLUSTER DICS
##############################################################################

def initial_cluster_dic(cluster_info_tsv, falcon_mgf, spectra_dic):
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
        pmz = row['precursor_mz']
        rt = row['retention_time']
        base = os.path.splitext(os.path.basename(fn))[0]
        sp_data = spectra_dic[(base, sc)]

        if cid not in cluster_dic:
            cluster_dic[cid] = {'scan_list': [], 'spec_pool': []}
        cluster_dic[cid]['scan_list'].append((fn, sc, pmz, rt))
        cluster_dic[cid]['spec_pool'].append(sp_data)

    # Process neg clusters (assign new IDs sequentially)
    for row in neg_rows:
        current_max_id += 1
        new_cid = current_max_id
        fn = row['filename']
        sc = int(row['scan'])
        pmz = row['precursor_mz']
        rt = row['retention_time']
        base = os.path.splitext(os.path.basename(fn))[0]
        sp_data = spectra_dic[(base, sc)]

        cluster_dic[new_cid] = {
            'scan_list': [(fn, sc, pmz, rt)],
            'spec_pool': [sp_data],
            'spectrum': sp_data  # Initial representative
        }

    # Attach Falcon MGF reps (use original IDs directly)
    mgf_spectra = read_mgf(falcon_mgf)
    for spectrum in mgf_spectra:
        original_cid = int(spectrum['cluster'])
        if original_cid in cluster_dic:
            cluster_dic[original_cid]['spectrum'] = spectrum
            cluster_dic[original_cid]['title'] = spectrum['title']
        else:
            print(f"Cluster {original_cid} not found, skipping.")

    return cluster_dic

def update_cluster_dic(cluster_dic, cluster_info_tsv, falcon_mgf, spectra_dic):
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
    for row in cat_noncons_nonneg:
        cid = int(row['cluster'])
        fn = row['filename']
        sc = int(row['scan'])
        pmz= row['precursor_mz']
        rt = row['retention_time']
        base = os.path.splitext(os.path.basename(fn))[0]
        sp_data = spectra_dic[(base, sc)]

        if cid not in currentID_uniID:
            new_key += 1
            currentID_uniID[cid] = new_key
            cluster_dic[new_key] = {'scan_list':[], 'spec_pool':[]}
        mapped_cid = currentID_uniID[cid]
        cluster_dic[mapped_cid]['scan_list'].append((fn, sc, pmz, rt))
        cluster_dic[mapped_cid]['spec_pool'].append(sp_data)

    # 3) cluster_id = -1
    new_cluster_id = max(cluster_dic.keys(), default=0)
    for row in cat_neg:
        fn = row['filename']
        sc = int(row['scan'])
        pmz= row['precursor_mz']
        rt = row['retention_time']
        base = os.path.splitext(os.path.basename(fn))[0]
        sp_data = spectra_dic[(base, sc)]
        new_cluster_id += 1
        currentID_uniID[new_cluster_id] = new_cluster_id
        cluster_dic[new_cluster_id] = {'scan_list':[], 'spec_pool':[]}
        cluster_dic[new_cluster_id]['scan_list'].append((fn, sc, pmz, rt))
        cluster_dic[new_cluster_id]['spec_pool'].append(sp_data)
        cluster_dic[new_cluster_id]['spectrum'] = sp_data

    # attach falcon mgf reps
    mgf_spectra = read_mgf(falcon_mgf)
    for s in mgf_spectra:
        try:
            old_cid = int(s['cluster'])
            if old_cid in currentID_uniID:
                new_cid = currentID_uniID[old_cid]
                cluster_dic[new_cid]['spectrum'] = s
                cluster_dic[new_cid]['title'] = s['title']
        except KeyError:
            print(old_cid, s['title'])

    return cluster_dic

def save_cluster_dic_optimized(cluster_dic, out_dir, max_workers=8):
    """Optimized saving with robust type handling"""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save scan_list with strict schema enforcement
    scan_schema = pa.schema([
        ("cluster_id", pa.int32()),
        ("filename", pa.string()),
        ("scan", pa.int32()),
        ("precursor_mz", pa.float32()),
        ("retention_time", pa.float32())
    ])

    def create_scan_chunk(chunk):
        data = []
        for cid, cdata in chunk:
            for fn, sc, pmz, rt in cdata.get("scan_list", []):
                data.append({
                    "cluster_id": int(cid),
                    "filename": str(fn),
                    "scan": int(sc),
                    "precursor_mz": float(pmz),
                    "retention_time": float(rt)
                })
        return pa.RecordBatch.from_pylist(data, schema=scan_schema)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunks = np.array_split(list(cluster_dic.items()), max_workers)
        record_batches = list(executor.map(create_scan_chunk, chunks))

        with pa.OSFile(os.path.join(out_dir, "scan_list.feather"), "wb") as sink:
            with pa.ipc.new_file(sink, scan_schema) as writer:
                for rb in record_batches:
                    if rb and len(rb) > 0:
                        writer.write(rb)

    # 2. Save spec_data with enhanced type validation
    spec_schema = pa.schema([
        ("cluster_id", pa.int32()),
        ("spec_pool_mz", pa.list_(pa.list_(pa.float32()))),
        ("spec_pool_intensity", pa.list_(pa.list_(pa.float32()))),
        ("spectrum_mz", pa.list_(pa.float32())),
        ("spectrum_intensity", pa.list_(pa.float32())),
        ("precursor_mz", pa.float32()),
        ("rtinseconds", pa.float32()),
        ("charge", pa.int32()),
        ("title", pa.string())
    ])

    def create_spec_entry(cid):
        cdata = cluster_dic[cid]
        spec = cdata.get("spectrum", {})

        # Robust charge handling
        raw_charge = str(spec.get("charge", "0")).strip('+').strip()
        charge = int(raw_charge) if raw_charge.isdigit() else 0

        return {
            "cluster_id": int(cid),
            "spec_pool_mz": [np.asarray(s["m/z array"], dtype=np.float32).tolist()
                             for s in cdata.get("spec_pool", [])],
            "spec_pool_intensity": [np.asarray(s["intensity array"], dtype=np.float32).tolist()
                                    for s in cdata.get("spec_pool", [])],
            "spectrum_mz": np.asarray(spec.get("m/z array", []), dtype=np.float32).tolist(),
            "spectrum_intensity": np.asarray(spec.get("intensity array", []), dtype=np.float32).tolist(),
            "precursor_mz": float(spec.get("precursor_mz", 0.0)),
            "rtinseconds": float(spec.get("rtinseconds", 0.0)),
            "charge": charge,
            "title": str(spec.get("title", ""))
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        spec_data = list(tqdm(executor.map(create_spec_entry, cluster_dic.keys()),
                              desc="Saving spec data", total=len(cluster_dic)))

        spec_table = pa.Table.from_pylist(spec_data, schema=spec_schema)
        pq.write_table(spec_table, os.path.join(out_dir, "spec_data.parquet"),
                       compression='ZSTD', use_dictionary=True)


def load_cluster_dic_optimized(in_dir, max_workers=8):
    """Optimized loading with complete data reconstruction"""
    cluster_dic = {}
    if not os.path.exists(in_dir):
        return cluster_dic

    # 1. Load and process scan_list
    scan_df = feather.read_feather(os.path.join(in_dir, "scan_list.feather"))

    def process_scan_chunk(chunk):
        return chunk.groupby('cluster_id')[['filename', 'scan', 'precursor_mz', 'retention_time']] \
            .apply(lambda x: x.values.tolist()) \
            .to_dict()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunks = np.array_split(scan_df, max_workers)
        for result in executor.map(process_scan_chunk, chunks):
            for cid, scans in result.items():
                cluster_dic.setdefault(cid, {}).setdefault("scan_list", []).extend(scans)

    # 2. Load and process spec_data
    spec_df = pq.read_table(os.path.join(in_dir, "spec_data.parquet")).to_pandas()

    def process_spec_row(row):
        # Convert charge back to string format
        charge = f"{row.charge}+" if row.charge > 0 else "0"

        return (row.cluster_id, {
            "spec_pool": [{"m/z array": mz, "intensity array": inten}
                          for mz, inten in zip(row.spec_pool_mz, row.spec_pool_intensity)],
            "spectrum": {
                "m/z array": row.spectrum_mz,
                "intensity array": row.spectrum_intensity,
                "precursor_mz": row.precursor_mz,
                "rtinseconds": row.rtinseconds,
                "charge": charge,
                "title": row.title
            }
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_spec_row, row) for row in spec_df.itertuples()]
        for future in concurrent.futures.as_completed(futures):
            cid, data = future.result()
            cluster_dic.setdefault(cid, {}).update(data)

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

    cluster_dic_path = os.path.join(checkpoint_dir, "cluster_dic.h5")
    consensus_path   = os.path.join(checkpoint_dir, "consensus.mzML")

    output_cluster_dic_path = os.path.join(output_dir, "cluster_dic.h5")
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
    print(f"[cluster_one_folder] Building spectra dict for folder: {folder}")
    folder_spec_dic = read_mzml_parallel(folder)

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
        cluster_dic = update_cluster_dic(cluster_dic, cluster_info_tsv, falcon_mgf_path, folder_spec_dic)
    else:
        # initial
        print("[cluster_one_folder] Performing initial clustering...")
        cluster_dic = initial_cluster_dic(cluster_info_tsv, falcon_mgf_path, folder_spec_dic)

    update_cluster_end_time = time.time()
    print(f"Merge results took {update_cluster_end_time - sumarize_results_end_time:.2f} s.")


    # Clean up local falcon outputs
    if os.path.exists("falcon.csv"):
        os.remove("falcon.csv")
    if os.path.exists("falcon.mgf"):
        os.remove("falcon.mgf")
    if os.path.exists("output_summary"):
        shutil.rmtree("output_summary", ignore_errors=True)

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

def finalize_results(cluster_dic, output_dir, current_batch_files=None):
    out_tsv = os.path.join(output_dir, "cluster_info.tsv")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ['precursor_mz','retention_time','cluster','filename','scan', 'new_batch']
    with open(out_tsv, 'w', newline='') as f:
        w = DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for cid, cdata in cluster_dic.items():
            for item in cdata['scan_list']:
                fn, sc, pmz, rt = item
                is_new = 'yes' if current_batch_files and os.path.basename(fn) in current_batch_files else 'no'
                row = {
                    'precursor_mz': pmz,
                    'retention_time': rt,
                    'cluster': cid,
                    'filename': fn,
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
    parser.add_argument("--max_mz", type=float, default=30000,
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
