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
import concurrent.futures
import io
from functools import lru_cache
from functools import partial
import gc


class MzMLIndexer:
    def __init__(self, folder_path):
        self.folder = folder_path
        self._index = {}  # Maps filename base to file path
        self._runs = {}  # Maps file path to open pymzml.run.Reader
        self._scan_id_types = {}  # Maps file path to 'str' or 'int'
        self._build_index()

    def _build_index(self):
        for fname in os.listdir(self.folder):
            if fname == "consensus.mzML" or not fname.endswith(".mzML"):
                continue
            file_path = os.path.join(self.folder, fname)
            base = os.path.splitext(fname)[0]
            self._index[base] = file_path

    @lru_cache(maxsize=100000)
    def get_spectrum(self, base, scan_id):
        scan_num = scan_id
        if not scan_num:
            return None

        file_path = self._index.get(base)
        if not file_path:
            return None

        # Open the file if not already open
        if file_path not in self._runs:
            try:
                run = pymzml.run.Reader(file_path, build_index=True, build_index_from_scratch=True)
                self._runs[file_path] = run
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
                return None

        run = self._runs[file_path]
        scan_type = self._scan_id_types.get(file_path)
        spectrum = None

        # Determine scan ID type if unknown
        if scan_type is None:
            try:
                spectrum = run[str(scan_num)]
                self._scan_id_types[file_path] = 'str'
            except Exception as e:
                try:
                    spectrum = run[int(scan_num)]
                    self._scan_id_types[file_path] = 'int'
                except Exception as e:
                    print(f"Error accessing {scan_num} in {file_path}: {e}")
        else:
            # Use known scan type
            if scan_type == 'str':
                try:
                    spectrum = run[str(scan_num)]
                except Exception as e:
                    try:
                        spectrum = run[int(scan_num)]
                    except Exception as e:
                        print(f"Error accessing stored type {scan_num} in {file_path}: {e}")
            elif scan_type == 'int':
                try:
                    spectrum = run[int(scan_num)]
                except Exception as e:
                    try:
                        spectrum = run[str(scan_num)]
                    except Exception as e:
                        print(f"Error accessing stored type {scan_num} in {file_path}: {e}")
        if not spectrum:
            print(f"Direct access for scan {scan_num} in {file_path} failed. Iterating over file to locate scan.")
            try:
                iteration_run = pymzml.run.Reader(file_path, build_index=True, build_index_from_scratch=True)
            except Exception as e:
                print(f"Error reopening {file_path} for iteration: {e}")
                return None
            for s in iteration_run:
                # Compare IDs as strings to be robust against type differences
                if str(s.get('id')) == str(scan_num):
                    spectrum = s
                    break
            if not spectrum:
                print(f"Scan {scan_num} not found in {file_path} even after iterating.")
                return None

        return self._create_spectrum_dict(spectrum)

    def _create_spectrum_dict(self, spectrum):
        precursor = spectrum.selected_precursors[0] if spectrum.selected_precursors else {}
        rt = spectrum.scan_time[0] if spectrum.scan_time else 0

        return {
            'peaks': list(spectrum.peaks("centroided")),
            'm/z array': spectrum.mz,
            'intensity array': spectrum.i,
            'precursor_mz': precursor.get('mz', 0),
            'rtinseconds': rt,
            'scans': spectrum['id'],
            'charge': precursor.get('charge', 0),
        }

    def close(self):
        for run in self._runs.values():
            try:
                run.close()
            except Exception:
                pass
        self._index.clear()
        self._runs.clear()
        self._scan_id_types.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_mzml_indexer(folder_path):
    """Create a lazy-loading indexer instead of loading all spectra"""
    return MzMLIndexer(folder_path)

def test_all_ms2_scans():
    base = "Blank_51"
    file_path = indexer._index.get(base)
    if not file_path:
        print(f"File for base {base} not found in the index.")
        return

    # Open the file using pymzml to iterate through all spectra
    try:
        run = pymzml.run.Reader(file_path, build_index=True, build_index_from_scratch=True)
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return

    # Collect all MS2 scan IDs
    ms2_scan_ids = []
    for spectrum in run:
        if spectrum.get('ms level') == 2:
            ms2_scan_ids.append(spectrum['id'])
    print(f"Found {len(ms2_scan_ids)} MS2 scans in {base}.")

    # Iterate through each MS2 scan and retrieve the spectrum using the indexer
    for scan_id in tqdm(ms2_scan_ids, desc="Processing MS2 scans"):
        sp = indexer.get_spectrum(base, scan_id)
        if sp is not None:
            pass
        else:
            print(f"Failed to retrieve spectrum for scan {scan_id}")

start_time = time.time()
indexer = create_mzml_indexer('/home/user/LabData/XianghuData/Test_incremental_400/batch_5/')
end_time = time.time()
print(f"Total time: {end_time - start_time}")
spectrum = indexer.get_spectrum("Blank_51","4427")
print(spectrum['scans'])
# print(spectrum['scans'])
#test_all_ms2_scans()


