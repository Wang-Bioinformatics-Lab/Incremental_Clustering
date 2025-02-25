#!/usr/bin/env python3

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
import gc

##############################################################################
# 1) FALCON + SUMMARIZE + MZML READ/WRITE FUNCTIONS
##############################################################################

def run_falcon(mzml_pattern, output_prefix="falcon"):
    """
    Runs Falcon on the given *.mzML pattern, specifying an output prefix (e.g. "falcon").
    Adjust the command arguments as needed.
    """
    command = (
        f"falcon {mzml_pattern} {output_prefix} "
        f"--export_representatives "
        f"--precursor_tol 20 ppm "
        f"--fragment_tol 0.05 "
        f"--min_mz_range 0 "
        f"--min_mz 0 --max_mz 30000 "
        f"--eps 0.1"
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
    for spectrum in tqdm(run, desc=f"Reading {os.path.basename(filepath)}", unit="spec"):
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
                # sample 9 from spec_pool + the 'spectrum'
                chosen = random.sample(data['spec_pool'], 9)
                chosen.append(data['spectrum'])
                for i, sp_data in enumerate(chosen, start=1):
                    new_id = f"{cid}_{i}"
                    _add_spectrum(exp, sp_data, new_id)
                write_count += 10
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
    Create a cluster_dic from scratch, given the output:
      - cluster_info.tsv from summarize_results.py
      - falcon.mgf from falcon
      - spectra_dic (the full dictionary of all ms2 spectra in the folder)
    """
    cluster_dic = {}
    with open(cluster_info_tsv, 'r') as csvfile:
        rdr = csv.DictReader(csvfile, delimiter='\t')
        for row in rdr:
            cid = int(row['cluster'])
            fn = row['filename']
            sc = int(row['scan'])
            pmz= row['precursor_mz']
            rt = row['retention_time']
            # find corresponding spec
            base = os.path.splitext(os.path.basename(fn))[0]
            sp_key = (base, sc)
            sp_data = spectra_dic[sp_key]
            if cid not in cluster_dic:
                cluster_dic[cid] = {'scan_list':[], 'spec_pool':[]}
            cluster_dic[cid]['scan_list'].append((fn, sc, pmz, rt))
            cluster_dic[cid]['spec_pool'].append(sp_data)

    # attach falcon mgf reps
    mgf_spectra = read_mgf(falcon_mgf)
    for s in mgf_spectra:
        c_id = int(s['cluster'])
        cluster_dic[c_id]['spectrum'] = s
        cluster_dic[c_id]['title'] = s['title']

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

##############################################################################
# 4) DRIVER LOGIC: ONE FOLDER AT A TIME
##############################################################################

def cluster_one_folder(folder, checkpoint_dir,output_dir,tool_dir):
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



    if os.path.exists(consensus_path):
        # Include the consensus file along with the new mzML files.
        input_files = f"{folder}/*.mzML {consensus_path}"
        print(f"[cluster_one_folder] Consensus file found; adding {consensus_path} to Falcon input.")
    else:
        input_files = f"{folder}/*.mzML"
        print(f"[cluster_one_folder] No consensus file found; proceeding with folder mzML files only.")

    # 1) RUN FALCON on the folder's *.mzML
    run_falcon(input_files, "falcon")

    # Build a dictionary of the spectra in this folder
    print(f"[cluster_one_folder] Building spectra dict for folder: {folder}")
    folder_spec_dic = read_mzml_parallel(folder)

    # Read existing cluster_dic (if any)
    cluster_dic = load_cluster_dic_hdf5_optimized(cluster_dic_path)
    has_checkpoint = (len(cluster_dic) > 0)

    # 2) Summarize => we get cluster_info.tsv
    cluster_info_tsv = summarize_output(output_dir,summarize_script= os.path.join(tool_dir,"summarize_results.py"), falcon_csv="falcon.csv")

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

    # Clean up local falcon outputs
    if os.path.exists("falcon.csv"):
        os.remove("falcon.csv")
    if os.path.exists("falcon.mgf"):
        os.remove("falcon.mgf")
    if os.path.exists("output_summary"):
        shutil.rmtree("output_summary", ignore_errors=True)

    # 4) Write updated consensus
    n_written = write_mzml(cluster_dic, output_consensus_path)
    print(f"[cluster_one_folder] Wrote {n_written} spectra to consensus: {consensus_path}")

    # 5) Save updated cluster_dic
    save_cluster_dic_hdf5_optimized(cluster_dic, output_cluster_dic_path)
    print(f"[cluster_one_folder] Updated cluster_dic saved at {cluster_dic_path}")

    finalize_results(cluster_dic, output_dir)

    # optional memory cleanup
    del cluster_dic, folder_spec_dic
    gc.collect()

def finalize_results(cluster_dic, output_dir):


    out_tsv = os.path.join(output_dir, "cluster_info.tsv")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ['precursor_mz','retention_time','cluster','filename','scan']
    with open(out_tsv, 'w', newline='') as f:
        w = DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for cid, cdata in cluster_dic.items():
            for item in cdata['scan_list']:
                fn, sc, pmz, rt = item
                row = {
                    'precursor_mz': pmz,
                    'retention_time': rt,
                    'cluster': cid,
                    'filename': fn,
                    'scan': sc
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
    args = parser.parse_args()

    # 1) Process exactly one folder
    cluster_one_folder(args.folder, args.checkpoint_dir, args.output_dir,args.tool_dir)


if __name__ == "__main__":
    main()
