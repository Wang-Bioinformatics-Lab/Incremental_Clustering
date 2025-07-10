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
import mmap
import struct
import sqlite3

##############################################################################
# 0) MEMORY-EFFICIENT SPECTRUM STORAGE SYSTEM
##############################################################################

class SpectrumStorage:
    """
    Memory-efficient spectrum storage using memory-mapped files and SQLite indexing.
    
    Features:
    - Memory-mapped binary storage for fast access
    - SQLite index for O(1) lookups
    - Minimal memory footprint
    - Efficient batch operations
    - Automatic file management
    """
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Binary storage file - directly in output directory, not in subfolder
        self.binary_file = self.storage_dir / "spectra.bin"
        self.index_db = self.storage_dir / "spectra_index.db"
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize binary file and SQLite index"""
        # Create binary file if not exists
        if not self.binary_file.exists():
            self.binary_file.write_bytes(b'')
        
        # Initialize SQLite index
        with sqlite3.connect(self.index_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spectrum_index (
                    filename TEXT,
                    scan INTEGER,
                    offset INTEGER,
                    size INTEGER,
                    precursor_mz REAL,
                    rtinseconds REAL,
                    charge INTEGER,
                    peak_count INTEGER,
                    PRIMARY KEY (filename, scan)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_filename_scan ON spectrum_index(filename, scan)")
            conn.commit()
    
    def _pack_spectrum(self, spectrum: dict) -> bytes:
        """Pack spectrum into binary format"""
        peaks = spectrum.get('peaks', [])
        peak_count = len(peaks)
        
        # Pack metadata
        metadata = struct.pack('<ddii', 
            spectrum.get('precursor_mz', 0.0),
            spectrum.get('rtinseconds', 0.0),
            spectrum.get('charge', 0),
            peak_count
        )
        
        # Pack peaks
        peak_data = b''
        for mz, intensity in peaks:
            peak_data += struct.pack('<ff', float(mz), float(intensity))
        
        return metadata + peak_data
    
    def _unpack_spectrum(self, data: bytes) -> dict:
        """Unpack spectrum from binary format"""
        # Unpack metadata
        metadata_size = struct.calcsize('<ddii')
        precursor_mz, rtinseconds, charge, peak_count = struct.unpack('<ddii', data[:metadata_size])
        
        # Unpack peaks
        peaks = []
        peak_size = struct.calcsize('<ff')
        for i in range(peak_count):
            offset = metadata_size + i * peak_size
            mz, intensity = struct.unpack('<ff', data[offset:offset + peak_size])
            peaks.append((mz, intensity))
        
        return {
            'peaks': peaks,
            'precursor_mz': precursor_mz,
            'rtinseconds': rtinseconds,
            'charge': charge,
            'title': ''
        }
    
    def store_spectrum(self, filename: str, scan: int, spectrum: dict):
        """Store a single spectrum"""
        # Pack spectrum
        spectrum_data = self._pack_spectrum(spectrum)
        
        # Append to binary file
        with open(self.binary_file, 'ab') as f:
            offset = f.tell()
            f.write(spectrum_data)
        
        # Update index
        with sqlite3.connect(self.index_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO spectrum_index 
                (filename, scan, offset, size, precursor_mz, rtinseconds, charge, peak_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(filename), int(scan), offset, len(spectrum_data),
                spectrum.get('precursor_mz', 0.0),
                spectrum.get('rtinseconds', 0.0),
                spectrum.get('charge', 0),
                len(spectrum.get('peaks', []))
            ))
            conn.commit()
    
    def store_spectra_batch(self, spectra_data: list):
        """Store multiple spectra efficiently with incremental update support (no deduplication)"""
        if not spectra_data:
            return
        
        # Use batch processing to avoid SQLite expression tree depth limit
        batch_size = 10000  # Process in smaller batches
        total_spectra = len(spectra_data)
        
        print(f"[store_spectra_batch] Processing {total_spectra} spectra in batches of {batch_size}")
        
        # Process in batches
        for batch_start in range(0, total_spectra, batch_size):
            batch_end = min(batch_start + batch_size, total_spectra)
            batch_data = spectra_data[batch_start:batch_end]
            print(f"[store_spectra_batch] Processing batch {batch_start//batch_size + 1}/{(total_spectra + batch_size - 1)//batch_size} ({len(batch_data)} spectra)")
            
            # Prepare data for all spectra in this batch
            index_data = []
            binary_data = b''
            spectrum_sizes = []
            
            for spec in batch_data:
                filename = spec['filename']
                scan = spec['scan']
                spectrum = {
                    'peaks': spec.get('peaks', []),
                    'precursor_mz': spec.get('precursor_mz', 0.0),
                    'rtinseconds': spec.get('rtinseconds', 0.0),
                    'charge': spec.get('charge', 0)
                }
                # Pack spectrum
                spectrum_data = self._pack_spectrum(spectrum)
                binary_data += spectrum_data
                spectrum_sizes.append(len(spectrum_data))
            
            # Write binary data and get global offsets
            with open(self.binary_file, 'ab') as f:
                file_start_offset = f.tell()
                f.write(binary_data)
            
            # Calculate global offsets for each spectrum
            offsets = []
            current_offset = file_start_offset
            for size in spectrum_sizes:
                offsets.append(current_offset)
                current_offset += size
            
            # Prepare index data for all spectra
            for i, spec in enumerate(batch_data):
                index_data.append((
                    str(spec['filename']), int(spec['scan']), offsets[i], spectrum_sizes[i],
                    spec.get('precursor_mz', 0.0),
                    spec.get('rtinseconds', 0.0),
                    spec.get('charge', 0),
                    len(spec.get('peaks', []))
                ))
            
            # Try INSERT first, fallback to INSERT OR REPLACE if there are conflicts
            with sqlite3.connect(self.index_db) as conn:
                try:
                    conn.executemany("""
                        INSERT INTO spectrum_index 
                        (filename, scan, offset, size, precursor_mz, rtinseconds, charge, peak_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, index_data)
                    conn.commit()
                except sqlite3.IntegrityError as e:
                    print(f"[store_spectra_batch] IntegrityError detected, using INSERT OR REPLACE: {e}")
                    conn.executemany("""
                        INSERT OR REPLACE INTO spectrum_index 
                        (filename, scan, offset, size, precursor_mz, rtinseconds, charge, peak_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, index_data)
                    conn.commit()
        
        print(f"[store_spectra_batch] Completed processing all {total_spectra} spectra")
    
    def get_spectrum(self, filename: str, scan: int) -> dict:
        """Get spectrum using memory-mapped access"""
        # Query index
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.execute("""
                SELECT offset, size FROM spectrum_index 
                WHERE filename = ? AND scan = ?
            """, (str(filename), int(scan)))
            result = cursor.fetchone()
            
            if not result:
                return None
            
            offset, size = result
        
        # Memory-mapped read
        with open(self.binary_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = mm[offset:offset + size]
                return self._unpack_spectrum(data)
    
    def get_spectra_batch(self, scan_list: list) -> dict:
        """Get multiple spectra efficiently"""
        if not scan_list:
            return {}
        
        # Use batch processing to avoid SQLite expression tree depth limit
        batch_size = 10000  # Process in smaller batches
        total_scans = len(scan_list)
        all_spectra = {}
        
        # Process in batches
        for batch_start in range(0, total_scans, batch_size):
            batch_end = min(batch_start + batch_size, total_scans)
            batch_scan_list = scan_list[batch_start:batch_end]
            
            # Query this batch
            conditions = []
            query_params = []
            for filename, scan in batch_scan_list:
                conditions.append("(filename = ? AND scan = ?)")
                query_params.extend([str(filename), int(scan)])
            
            where_clause = " OR ".join(conditions)
            
            with sqlite3.connect(self.index_db) as conn:
                cursor = conn.execute(f"""
                    SELECT filename, scan, offset, size FROM spectrum_index 
                    WHERE {where_clause}
                """, query_params)
                results = cursor.fetchall()
            
            # Memory-mapped batch read for this batch
            if results and self.binary_file.stat().st_size > 0:
                with open(self.binary_file, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        for filename, scan, offset, size in results:
                            data = mm[offset:offset + size]
                            spectrum = self._unpack_spectrum(data)
                            all_spectra[(filename, scan)] = spectrum
        
        if not all_spectra:
            print(f"[Warning] No spectra found in storage or binary file is empty")
        
        return all_spectra
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics"""
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM spectrum_index")
            count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT SUM(size) FROM spectrum_index")
            total_size = cursor.fetchone()[0] or 0
        
        return {
            'spectrum_count': count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'binary_file_size_mb': self.binary_file.stat().st_size / (1024 * 1024)
        }
    
    def cleanup_storage(self):
        """Clean up storage by removing orphaned data and compacting binary file"""
        print("[cleanup_storage] Starting storage cleanup...")
        
        # Get all valid offsets and sizes
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.execute("""
                SELECT offset, size FROM spectrum_index 
                ORDER BY offset
            """)
            valid_regions = cursor.fetchall()
        
        if not valid_regions:
            print("[cleanup_storage] No data to clean up")
            return
        
        # Create new compacted binary file
        temp_binary = self.binary_file.with_suffix('.tmp')
        new_offsets = {}
        current_offset = 0
        
        with open(self.binary_file, 'rb') as old_file, open(temp_binary, 'wb') as new_file:
            for old_offset, size in valid_regions:
                # Read data from old file
                old_file.seek(old_offset)
                data = old_file.read(size)
                
                # Write to new file
                new_file.write(data)
                
                # Record new offset
                new_offsets[old_offset] = current_offset
                current_offset += size
        
        # Update database with new offsets
        with sqlite3.connect(self.index_db) as conn:
            for old_offset, new_offset in new_offsets.items():
                conn.execute("""
                    UPDATE spectrum_index 
                    SET offset = ? 
                    WHERE offset = ?
                """, (new_offset, old_offset))
            conn.commit()
        
        # Replace old file with new compacted file
        self.binary_file.unlink()
        temp_binary.rename(self.binary_file)
        
        # Get new stats
        stats = self.get_storage_stats()
        print(f"[cleanup_storage] Cleanup completed. New size: {stats['binary_file_size_mb']:.1f}MB")
    
    def remove_spectra(self, scan_list: list):
        """Remove specific spectra from storage"""
        if not scan_list:
            return
        
        # Build batch delete query
        conditions = []
        query_params = []
        for filename, scan in scan_list:
            conditions.append("(filename = ? AND scan = ?)")
            query_params.extend([str(filename), int(scan)])
        
        where_clause = " OR ".join(conditions)
        
        with sqlite3.connect(self.index_db) as conn:
            # Delete from index
            cursor = conn.execute(f"""
                DELETE FROM spectrum_index 
                WHERE {where_clause}
            """, query_params)
            deleted_count = cursor.rowcount
            conn.commit()
        
        print(f"[remove_spectra] Removed {deleted_count} spectra from index")
        
        # Note: Binary file cleanup should be done separately with cleanup_storage()
        # to avoid fragmentation

# Global spectrum storage instances cache
_spectrum_storage_instances = {}

def get_spectrum_storage(storage_dir: str = None) -> SpectrumStorage:
    """Get or create spectrum storage instance for specific directory"""
    if storage_dir is None:
        storage_dir = "spectrum_storage"
    
    # Use absolute path as key to avoid conflicts
    abs_storage_dir = os.path.abspath(storage_dir)
    
    # Check if instance already exists for this directory
    if abs_storage_dir not in _spectrum_storage_instances:
        _spectrum_storage_instances[abs_storage_dir] = SpectrumStorage(abs_storage_dir)
    
    return _spectrum_storage_instances[abs_storage_dir]

def migrate_old_storage_if_needed(checkpoint_dir: str):
    """Migrate old storage format to new format if needed"""
    old_index = os.path.join(checkpoint_dir, "spectra_index.db")
    old_binary = os.path.join(checkpoint_dir, "spectra.bin")
    new_storage_dir = os.path.join(checkpoint_dir, "spectrum_storage")
    new_index = os.path.join(new_storage_dir, "spectra_index.db")
    new_binary = os.path.join(new_storage_dir, "spectra.bin")
    
    # Check if old format exists and new format is empty
    if (os.path.exists(old_index) and os.path.exists(old_binary) and 
        os.path.getsize(old_binary) > 0 and 
        (not os.path.exists(new_binary) or os.path.getsize(new_binary) == 0)):
        
        print(f"[Info] Migrating old storage format to new format...")
        
        # Create new storage directory
        os.makedirs(new_storage_dir, exist_ok=True)
        
        # Copy files
        import shutil
        shutil.copy2(old_index, new_index)
        shutil.copy2(old_binary, new_binary)
        
        print(f"[Info] Migration completed: {os.path.getsize(old_binary)} bytes moved to {new_storage_dir}")
        
        # Verify migration
        if os.path.exists(new_binary) and os.path.getsize(new_binary) > 0:
            print(f"[Info] Migration verification successful")
        else:
            print(f"[Warning] Migration verification failed")



##############################################################################
# 1) FALCON + SUMMARIZE + MZML READ/WRITE FUNCTIONS
##############################################################################

def parse_scan_identifier(scan_id):
    """Parse scan identifier that may contain full path with underscore and scan number"""
    if isinstance(scan_id, str) and '_' in scan_id:
        # Try to find the last occurrence of .mzML_ followed by numbers
        import re
        match = re.search(r'\.mzML_(\d+)$', scan_id)
        if match:
            scan_number = int(match.group(1))
            file_path = scan_id[:-len(match.group(0))] + '.mzML'
            return file_path, scan_number
    
    # Fallback: assume scan_id is already a scan number
    return None, int(scan_id) if isinstance(scan_id, (int, str)) else scan_id

def is_in_current_batch(fp, current_batch_folder):
    return os.path.commonpath([os.path.abspath(fp), os.path.abspath(current_batch_folder)]) == os.path.abspath(current_batch_folder)

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

def get_original_file_path(filename, original_filepah):
    # Return absolute path to ensure consistency with database storage
    return os.path.abspath(os.path.join(original_filepah, filename))

def read_mzml(filepath):
    """
    Reads MS2 spectra from an mzML file via pymzml, returns list of spectrum dicts.
    """
    spectra = []
    try:
        run = pymzml.run.Reader(filepath, build_index_from_scratch=True)
        for spectrum in run:
            if spectrum['ms level'] == 2:
                try:
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
                except Exception as e:
                    print(f"[Warning] Error processing spectrum in {filepath}: {e}")
                    continue
    except Exception as e:
        print(f"[Error] Failed to read mzML file {filepath}: {e}")
        raise
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


def get_spectrum_from_storage(fp, sc, checkpoint_dir):
    """Get spectrum using memory-efficient storage system"""
    # First try to get consensus spectrum (still use parquet for consensus)
    consensus_path = os.path.join(checkpoint_dir, "consensus.parquet")
    if os.path.exists(consensus_path):
        # Check if this is a consensus spectrum (filename format: consensus_xxx)
        if str(fp).startswith("consensus_"):
            try:
                cluster_id = int(str(fp).split("_")[1])
                consensus = get_consensus_from_parquet(cluster_id, consensus_path)
                if consensus:
                    return consensus
            except (ValueError, IndexError):
                pass
    
    # Try memory-efficient storage
    storage_dir = os.path.join(checkpoint_dir, "spectrum_storage")
    storage = get_spectrum_storage(storage_dir)
    spectrum = storage.get_spectrum(str(fp), int(sc))
    if spectrum is not None:
        return spectrum
    
    return None

def write_singletons_mzml(scan_list, output_path, current_batch_folder, checkpoint_dir):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    exp = oms.MSExperiment()
    current_batch_scans = []
    history_scans = []
    for scan_item in scan_list:
        if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
            fp, sc, precursor_mz, retention_time = scan_item
        else:  # Legacy format (filename, scan)
            fp, sc = scan_item
        if is_in_current_batch(fp, current_batch_folder):
            current_batch_scans.append((fp, sc))
        else:
            history_scans.append((fp, sc))

    # Group by file, one process handles all scans from one file
    if current_batch_scans:
        file_to_scans = defaultdict(list)
        for fp, sc in current_batch_scans:
            file_to_scans[fp].append(sc)
        from joblib import Parallel, delayed
        def fetch_file_scans(fp, scans):
            results = []
            if not os.path.exists(fp):
                print(f"[Warning] File not found: {fp}, skipping scans: {scans}")
                return results
            try:
                spectra = read_mzml(fp)
                scan_to_spectrum = {int(s['scans']): s for s in spectra}
                for sc in scans:
                    scan_id = int(sc)
                    if scan_id in scan_to_spectrum:
                        s = scan_to_spectrum[scan_id]
                        # construct unique scan_id: absolute path + underscore + scan number
                        unique_scan = f"{os.path.abspath(fp)}_{scan_id}"
                        s = dict(s)  # copy, avoid polluting original data
                        s['unique_scan'] = unique_scan
                        results.append((unique_scan, s))
            except Exception as e:
                print(f"[Error] Failed to read file {fp}: {e}")
            return results
        available_cores = os.cpu_count() or 1
        N_JOBS = max(40, min(available_cores // 2, 96))
        all_results = Parallel(n_jobs=N_JOBS)(
            delayed(fetch_file_scans)(fp, scans) for fp, scans in file_to_scans.items()
        )
        for file_results in all_results:
            for unique_scan, s in file_results:
                _add_spectrum(exp, s, unique_scan)

    # Use efficient batch lookup for historical scans
    if history_scans:
        storage = get_spectrum_storage(checkpoint_dir)  # Checkpoint storage is in checkpoint_dir directly
        spectra_batch = storage.get_spectra_batch(history_scans)
        print(f"[write_singletons_mzml] Batch loaded {len(spectra_batch)}/{len(history_scans)} historical spectra")
        
        for fp, sc in history_scans:
            sp = spectra_batch.get((str(fp), int(sc)))
            if sp is not None:
                unique_scan = f"{os.path.abspath(fp)}_{sc}"
                sp = dict(sp)
                sp['unique_scan'] = unique_scan
                _add_spectrum(exp, sp, unique_scan)
            else:
                print(f"[Warning] Cannot find spectrum for {fp}, scan {sc} in checkpoint!")
    
    oms.MzMLFile().store(str(output_path), exp)
    print(f"[write_singletons_mzml] Wrote {exp.size()} spectra to {output_path}")
    
    # Print storage statistics if storage was used
    if history_scans:
        stats = storage.get_storage_stats()
        print(f"[write_singletons_mzml] Storage stats: {stats}")


def write_mzml(cluster_dic, out_path, current_batch_folder, checkpoint_dir, sample_threshold=5):
    # Ensure output directory exists
    output_dir = os.path.dirname(out_path)
    if output_dir:  # Only create directory if there is a directory component
        os.makedirs(output_dir, exist_ok=True)
    print(f"[Debug] write_mzml - out_path: {out_path}")
    print(f"[Debug] write_mzml - output_dir: {output_dir}")
    exp = oms.MSExperiment()
    current_batch_scans = []
    history_scans = []
    
    for cid, data in cluster_dic.items():
        scan_list = data.get('scan_list', [])
        sample_list = data.get('sample_list', [])
        
        # Generate sample_list for large clusters if not exists
        if len(scan_list) > sample_threshold and not sample_list:
            # Large cluster without sample_list, need to sample
            sample_list = random.sample(scan_list, sample_threshold)
            data['sample_list'] = sample_list  # update cluster_dic
        
        # Always add consensus spectrum if exists
        if 'spectrum' in data:
            _add_spectrum(exp, data['spectrum'], cid)
        
        # For large clusters, also add sample_list spectra
        if len(scan_list) > sample_threshold and sample_list:
            for i, scan_item in enumerate(sample_list):
                if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
                    fp, sc, precursor_mz, retention_time = scan_item
                else:  # Legacy format (filename, scan)
                    fp, sc = scan_item
                out_id = f"{cid}_{i+1}"
                if is_in_current_batch(fp, current_batch_folder):
                    current_batch_scans.append((fp, sc, out_id))
                else:
                    history_scans.append((fp, sc, out_id))

    # Group by file, one process handles all scans from one file
    if current_batch_scans:
        file_to_scans = defaultdict(list)
        for fp, sc, out_id in current_batch_scans:
            file_to_scans[fp].append((sc, out_id))
        
        from joblib import Parallel, delayed
        def fetch_file_scans(fp, scans):
            results = []
            # Check if file exists before trying to read it
            if not os.path.exists(fp):
                print(f"[Warning] File not found: {fp}, skipping scans: {scans}")
                return results
            try:
                spectra = read_mzml(fp)
                # Create a mapping from scan_id to spectrum for O(1) lookup
                scan_to_spectrum = {int(s['scans']): s for s in spectra}
                for sc, out_id in scans:
                    scan_id = int(sc)
                    if scan_id in scan_to_spectrum:
                        results.append((out_id, scan_to_spectrum[scan_id]))
            except Exception as e:
                print(f"[Error] Failed to read file {fp}: {e}")
            return results
        
        # Maintain original parallel count
        available_cores = os.cpu_count() or 1
        N_JOBS = max(40, min(available_cores // 2, 96))
        
        all_results = Parallel(n_jobs=N_JOBS)(
            delayed(fetch_file_scans)(fp, scans) for fp, scans in file_to_scans.items()
        )
        for file_results in all_results:
            for out_id, s in file_results:
                _add_spectrum(exp, s, out_id)

    # Use efficient batch lookup for historical scans
    if history_scans:
        storage = get_spectrum_storage(checkpoint_dir)  # Checkpoint storage is in checkpoint_dir directly
        history_scan_list = [(fp, sc) for fp, sc, _ in history_scans]
        spectra_batch = storage.get_spectra_batch(history_scan_list)
        print(f"[write_mzml] Batch loaded {len(spectra_batch)}/{len(history_scan_list)} historical spectra")
        
        for fp, sc, out_id in history_scans:
            sp = spectra_batch.get((str(fp), int(sc)))
            if sp is not None:
                _add_spectrum(exp, sp, out_id)
            else:
                print(f"[Warning] Cannot find spectrum for {fp}, scan {sc} in checkpoint!")
    
    oms.MzMLFile().store(str(out_path), exp)
    print(f"[write_mzml] Wrote {exp.size()} spectra to {out_path}")
    
    # Print storage statistics if storage was used
    if history_scans:
        stats = storage.get_storage_stats()
        print(f"[write_mzml] Storage stats: {stats}")


def _process_fetch_file(file_path, scan_pairs, temp_dir, current_batch_folder, checkpoint_dir):
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
    ch_str = str(sp_data.get('charge', sp_data.get('CHARGE', '0'))).replace('+','')
    if ch_str.isdigit():
        prec.setCharge(int(ch_str))
    else:
        prec.setCharge(0)
    s.setPrecursors([prec])
    # Use unique scan_id
    unique_id = sp_data.get('unique_scan', scan_id)
    s.setNativeID(f"controllerType=0 controllerNumber=1 scan={unique_id}")
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
        cid = int(row['cluster']) + 1 # cluster id starts from 1
        current_max_id = max(current_max_id, cid)
        fn = row['filename']
        scan_str = row['scan']
        
        # Parse scan identifier to handle cases where scan contains full path
        parsed_fp, parsed_sc = parse_scan_identifier(scan_str)
        if parsed_fp is not None:
            # scan contained full path, use parsed values
            fp = parsed_fp
            sc = parsed_sc
        else:
            # scan is just a scan number, use original filename
            sc = parsed_sc
            fp = get_original_file_path(fn, original_file_path)
        
        # Extract precursor_mz and retention_time from Falcon CSV data
        precursor_mz = float(row.get('precursor_mz', 0))
        retention_time = float(row.get('retention_time', 0))
        
        if cid not in cluster_dic:
            cluster_dic[cid] = {'scan_list': []}
        cluster_dic[cid]['scan_list'].append((fp, sc, precursor_mz, retention_time))

    # Process neg clusters (singletons)
    for row in neg_rows:
        fn = row['filename']
        scan_str = row['scan']
        
        # Parse scan identifier to handle cases where scan contains full path
        parsed_fp, parsed_sc = parse_scan_identifier(scan_str)
        if parsed_fp is not None:
            # scan contained full path, use parsed values
            fp = parsed_fp
            sc = parsed_sc
        else:
            # scan is just a scan number, use original filename
            sc = parsed_sc
            fp = get_original_file_path(fn, original_file_path)
        
        # Extract precursor_mz and retention_time from Falcon CSV data
        precursor_mz = float(row.get('precursor_mz', 0))
        retention_time = float(row.get('retention_time', 0))
        
        singletons.append((fp, sc, precursor_mz, retention_time))

    # Attach Falcon MGF reps
    mgf_spectra = read_mgf(falcon_mgf)
    for spectrum in mgf_spectra:
        original_cid = int(spectrum['cluster'])
        new_cid = original_cid + 1
        if new_cid in cluster_dic:
            cluster_dic[new_cid]['spectrum'] = spectrum

    return cluster_dic, singletons

def update_cluster_dic(cluster_dic, cluster_info_tsv, falcon_mgf, original_file_path):
    """
    Merge new clustering results into existing cluster_dic, returning:
    - Updated cluster_dic (only non-singletons)
    - List of new singletons (filepath, scan, precursor_mz, retention_time) tuples
    """
    currentID_uniID = {}  # Mapping from Falcon's temporary IDs to unified cluster IDs
    new_singletons = []  # Stores (filepath, scan, precursor_mz, retention_time) of new singletons
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
                # Handle singletons
                scan_str = row['scan']
                # Parse scan identifier to handle cases where scan contains full path
                parsed_fp, parsed_sc = parse_scan_identifier(scan_str)
                if parsed_fp is not None:
                    # scan contained full path, use parsed values
                    fp = parsed_fp
                    sc = parsed_sc
                else:
                    # scan is just a scan number, use original filename
                    sc = parsed_sc
                    fp = get_original_file_path(row['filename'], original_file_path)
                
                # Extract precursor_mz and retention_time from Falcon CSV data
                precursor_mz = float(row.get('precursor_mz', 0))
                retention_time = float(row.get('retention_time', 0))
                
                new_singletons.append((fp, sc, precursor_mz, retention_time))

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
        
        # Parse scan identifier to handle cases where scan contains full path
        parsed_fp, parsed_sc = parse_scan_identifier(scan_str)
        if parsed_fp is not None:
            # scan contained full path, use parsed values
            fp = parsed_fp
            sc = parsed_sc
        else:
            # scan is just a scan number, use original filename
            sc = parsed_sc
            fp = get_original_file_path(fn, original_file_path)
        
        # Extract precursor_mz and retention_time from Falcon CSV data
        precursor_mz = float(row.get('precursor_mz', 0))
        retention_time = float(row.get('retention_time', 0))
        
        # Get or create unified cluster ID
        if cid not in currentID_uniID:
            new_cluster_id += 1
            currentID_uniID[cid] = new_cluster_id
            cluster_dic[new_cluster_id] = {'scan_list':[], 'spec_pool':[]}
        mapped_cid = currentID_uniID[cid]
        cluster_dic[mapped_cid]['scan_list'].append((fp, sc, precursor_mz, retention_time))

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


def save_cluster_dic_optimized(cluster_dic, out_dir, singletons=None, singletons_mzml_path=None, current_batch_folder=None, sample_threshold=5):
    """
    Save cluster_dic structure to feather files (scan_list and sample_list only).
    Spectrum bodies are managed entirely by the efficient storage system, only save indices here.
    """
    os.makedirs(out_dir, exist_ok=True)
    scan_data = []
    sample_data = []

    for cid, cdata in cluster_dic.items():
        # Only save scan_list information
        for scan_item in cdata.get("scan_list", []):
            if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
                fp, sc, precursor_mz, retention_time = scan_item
                scan_data.append({
                    "cluster_id": cid,
                    "filename": fp,
                    "scan": sc,
                    "precursor_mz": precursor_mz,
                    "retention_time": retention_time
                })
            else:  # Legacy format (filename, scan)
                fp, sc = scan_item
                scan_data.append({
                    "cluster_id": cid,
                    "filename": fp,
                    "scan": sc,
                    "precursor_mz": 0.0,
                    "retention_time": 0.0
                })
        # Only save sample_list information
        sample_list = cdata.get("sample_list", [])
        if sample_list:
            for scan_item in sample_list:
                if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
                    fp, sc, precursor_mz, retention_time = scan_item
                    sample_data.append({
                        "cluster_id": cid,
                        "filename": fp,
                        "scan": sc,
                        "precursor_mz": precursor_mz,
                        "retention_time": retention_time
                    })
                else:  # Legacy format (filename, scan)
                    fp, sc = scan_item
                    sample_data.append({
                        "cluster_id": cid,
                        "filename": fp,
                        "scan": sc,
                        "precursor_mz": 0.0,
                        "retention_time": 0.0
                    })

    # Save scan_list.feather
    scan_table = pa.Table.from_pylist(scan_data)
    feather.write_feather(scan_table, os.path.join(out_dir, "scan_list.feather"))
    # Save sample_list.feather
    if sample_data:
        sample_table = pa.Table.from_pylist(sample_data)
        feather.write_feather(sample_table, os.path.join(out_dir, "sample_list.feather"))
    # Consensus spectra are saved separately in cluster_one_folder
    # to avoid duplicate saving
    # Initialize efficient storage - use output directory directly, don't create subfolder
    get_spectrum_storage(out_dir)
    print(f"[save_cluster_dic_optimized] Saved {len(scan_data)} scan_list entries and {len(sample_data)} sample_list entries (no spectrum bodies)")

def load_cluster_dic_optimized(in_dir):
    """
    Only load scan_list and sample_list, spectrum bodies are no longer redundantly loaded from disk.
    When spectrum is needed, dynamically look it up through consensus.parquet or efficient storage system.
    Automatically supplement consensus spectrum for each cluster during loading (if available).
    Returns: (cluster_dic, max_cluster_id)
    """
    cluster_dic = {}
    max_cluster_id = 0
    if not os.path.exists(in_dir):
        return cluster_dic, max_cluster_id
    # Load scan_list
    scan_path = os.path.join(in_dir, "scan_list.feather")
    if os.path.exists(scan_path):
        scan_df = feather.read_table(scan_path).to_pandas()
        for row in scan_df.itertuples():
            cluster_dic.setdefault(row.cluster_id, {'scan_list': []})
            # Check if precursor_mz and retention_time columns exist
            if hasattr(row, 'precursor_mz') and hasattr(row, 'retention_time'):
                cluster_dic[row.cluster_id]['scan_list'].append((row.filename, row.scan, row.precursor_mz, row.retention_time))
            else:
                # Legacy format - use default values
                cluster_dic[row.cluster_id]['scan_list'].append((row.filename, row.scan, 0.0, 0.0))
    # Load sample_list
    sample_path = os.path.join(in_dir, "sample_list.feather")
    if os.path.exists(sample_path):
        sample_df = feather.read_table(sample_path).to_pandas()
        for row in sample_df.itertuples():
            if row.cluster_id in cluster_dic:
                cluster_dic[row.cluster_id].setdefault('sample_list', [])
                # Check if precursor_mz and retention_time columns exist
                if hasattr(row, 'precursor_mz') and hasattr(row, 'retention_time'):
                    cluster_dic[row.cluster_id]['sample_list'].append((row.filename, row.scan, row.precursor_mz, row.retention_time))
                else:
                    # Legacy format - use default values
                    cluster_dic[row.cluster_id]['sample_list'].append((row.filename, row.scan, 0.0, 0.0))
    # Automatically supplement consensus spectrum and track max cluster ID
    consensus_path = os.path.join(in_dir, "consensus.parquet")
    if os.path.exists(consensus_path):
        consensus_df = pq.read_table(consensus_path).to_pandas()
        for row in consensus_df.itertuples():
            max_cluster_id = max(max_cluster_id, row.cluster_id)
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
    
    # Also check scan_list for max cluster ID
    if cluster_dic:
        max_cluster_id = max(max_cluster_id, max(cluster_dic.keys()))
    
    return cluster_dic, max_cluster_id

def _custom_merge_mzml_files(file_list, output_path, threads=8):
    """
    Legacy custom mzML merger (kept for compatibility).
    Use _custom_merge_mzml_files_parallel for better performance.
    """
    return _custom_merge_mzml_files_parallel(file_list, output_path, threads)

##############################################################################
# 4) DRIVER LOGIC: ONE FOLDER AT A TIME
##############################################################################

def collect_scans_for_next_batch(cluster_dic, current_batch_folder, phase2_singletons=None):
    """Collect scans that need to be saved for the next batch"""
    scans_to_save = set()
    
    # 1. Collect from write_mzml: current batch scans in newly created sample_list
    for cid, data in cluster_dic.items():
        scan_list = data.get('scan_list', [])
        sample_list = data.get('sample_list', [])
        
        if len(scan_list) > 5 and sample_list:
            # Large cluster, collect current batch scans from sample_list
            for scan_item in sample_list:
                if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
                    fp, sc, precursor_mz, retention_time = scan_item
                else:  # Legacy format (filename, scan)
                    fp, sc = scan_item
                if is_in_current_batch(fp, current_batch_folder):
                    scans_to_save.add((fp, sc))
        elif len(scan_list) <= 5:
            # Small cluster, collect current batch scans from scan_list
            # These might be needed for sampling in future rounds if cluster becomes large
            for scan_item in scan_list:
                if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
                    fp, sc, precursor_mz, retention_time = scan_item
                else:  # Legacy format (filename, scan)
                    fp, sc = scan_item
                if is_in_current_batch(fp, current_batch_folder):
                    scans_to_save.add((fp, sc))
    
    # 2. Collect from write_singletons_mzml: current batch scans in phase 2 singletons
    if phase2_singletons:
        for scan_item in phase2_singletons:
            if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
                fp, sc, precursor_mz, retention_time = scan_item
            else:  # Legacy format (filename, scan)
                fp, sc = scan_item
            if is_in_current_batch(fp, current_batch_folder):
                scans_to_save.add((fp, sc))
    
    return list(scans_to_save)


# Removed: save_spectra_incremental - replaced by memory-efficient storage system


# Removed: save_spectra_batch - replaced by memory-efficient storage system


def save_consensus_incremental(cluster_dic, consensus_parquet_path, max_existing_cluster_id=None):
    """Incrementally save consensus spectra - optimized for incremental updates"""
    consensus_data = []
    
    # If max_existing_cluster_id is provided, only process new clusters
    # This avoids checking existence for each cluster
    for cid, cdata in cluster_dic.items():
        # Skip if this cluster ID is not new (if max_existing_cluster_id is provided)
        if max_existing_cluster_id is not None and cid <= max_existing_cluster_id:
            continue
            
        spectrum = cdata.get("spectrum")
        if spectrum and 'peaks' in spectrum and len(spectrum['peaks']) > 0:
            peaks = np.array(spectrum['peaks'], dtype=np.float32)
            ch_str = str(spectrum.get('charge', '0')).replace('+', '')
            charge_val = int(ch_str) if ch_str.isdigit() else 0
            
            consensus_data.append({
                "cluster_id": cid,
                "filename": f"consensus_{cid}",
                "scan": int(cid),
                "spectrum_mz": peaks[:, 0].tobytes(),
                "spectrum_intensity": peaks[:, 1].tobytes(),
                "precursor_mz": np.float32(spectrum.get('precursor_mz', 0)),
                "rtinseconds": np.float32(spectrum.get('rtinseconds', 0)),
                "charge": np.int32(charge_val),
                "title": str(spectrum.get('title', ''))
            })
    
    # Incrementally write
    if consensus_data:
        print(f"[save_consensus_incremental] Adding {len(consensus_data)} new consensus spectra")
        save_consensus_batch(consensus_data, consensus_parquet_path)
    else:
        print(f"[save_consensus_incremental] No new consensus spectra to add")


def get_max_existing_cluster_id(consensus_parquet_path):
    """Get the maximum cluster ID from existing consensus file"""
    if not os.path.exists(consensus_parquet_path):
        return 0
    
    try:
        table = pq.read_table(consensus_parquet_path)
        if table.num_rows == 0:
            return 0
        
        # Convert to pandas to get max value
        df = table.to_pandas()
        max_id = df['cluster_id'].max()
        return int(max_id) if max_id is not None else 0
    except Exception as e:
        print(f"[Warning] Failed to get max cluster ID from {consensus_parquet_path}: {e}")
        return 0


def consensus_exists(cluster_id, consensus_parquet_path):
    """Check if consensus for a cluster already exists"""
    if not os.path.exists(consensus_parquet_path):
        return False
    
    try:
        table = pq.read_table(
            consensus_parquet_path,
            filters=[('cluster_id', '=', cluster_id)]
        )
        return table.num_rows > 0
    except Exception:
        return False


def save_consensus_batch(consensus_data, consensus_parquet_path):
    """Save a batch of consensus spectra to parquet file"""
    schema = pa.schema([
        ('cluster_id', pa.int32()),
        ('filename', pa.string()),
        ('scan', pa.int32()),
        ('spectrum_mz', pa.binary()),
        ('spectrum_intensity', pa.binary()),
        ('precursor_mz', pa.float32()),
        ('rtinseconds', pa.float32()),
        ('charge', pa.int32()),
        ('title', pa.string())
    ])
    
    # Check if file exists for incremental append
    if os.path.exists(consensus_parquet_path):
        # Read existing data to avoid duplicates
        existing_df = pq.read_table(consensus_parquet_path).to_pandas()
        existing_clusters = set(existing_df['cluster_id'])
        
        # Filter out duplicates
        new_consensus = []
        for consensus in consensus_data:
            if consensus['cluster_id'] not in existing_clusters:
                new_consensus.append(consensus)
                existing_clusters.add(consensus['cluster_id'])
        
        if new_consensus:
            new_table = pa.Table.from_pylist(new_consensus, schema=schema)
            # Append to existing file
            existing_table = pa.Table.from_pandas(existing_df)
            combined_table = pa.concat_tables([existing_table, new_table])
            pq.write_table(
                combined_table,
                consensus_parquet_path,
                compression='zstd',
                compression_level=3
            )
    else:
        # Create new file
        consensus_table = pa.Table.from_pylist(consensus_data, schema=schema)
        pq.write_table(
            consensus_table,
            consensus_parquet_path,
            compression='zstd',
            compression_level=3
        )


# Removed: get_spectrum_from_spectra_parquet - replaced by memory-efficient storage


def get_consensus_from_parquet(cluster_id, consensus_parquet_path):
    """Get consensus spectrum from consensus.parquet"""
    if not os.path.exists(consensus_parquet_path):
        return None
    
    try:
        table = pq.read_table(
            consensus_parquet_path,
            filters=[('cluster_id', '=', cluster_id)]
        )
        if table.num_rows == 0:
            return None
        
        row = table.to_pandas().iloc[0]
        mz = np.frombuffer(row['spectrum_mz'], dtype=np.float32)
        intensity = np.frombuffer(row['spectrum_intensity'], dtype=np.float32)
        return {
            'peaks': list(zip(mz, intensity)),
            'precursor_mz': row['precursor_mz'],
            'rtinseconds': row['rtinseconds'],
            'charge': row['charge'],
            'title': row['title']
        }
    except Exception as e:
        print(f"[Warning] Failed to read consensus from {consensus_parquet_path}: {e}")
        return None


def cluster_one_folder(folder, checkpoint_dir, output_dir, tool_dir, precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps):
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

    consensus_path   = os.path.join(checkpoint_dir, "consensus.mzML")

    output_consensus_path = os.path.join(output_dir, "consensus.mzML")

    scan_feather = os.path.join(checkpoint_dir, "scan_list.feather")
    storage_bin = os.path.join(checkpoint_dir, "spectra.bin")

    has_checkpoint = os.path.exists(scan_feather) and os.path.exists(storage_bin)

    if not has_checkpoint:
        #run falcon
        print("[Initial] Running Falcon on all new spectra...")
        input_files = f"{folder}/*.mzML"
        run_falcon(input_files, "falcon", precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps)
        falcon_end_time = time.time()
        timing_log['Falcon initial'] = falcon_end_time - start_time
        print(f"Falcon clustering took {falcon_end_time - start_time:.2f} s.")

        #summarize results
        cluster_info_tsv = summarize_output(output_dir, os.path.join(tool_dir, 'summarize_results.py'), 'falcon.csv')
        summarize_time = time.time()
        timing_log['summarize initial results'] = summarize_time - falcon_end_time
        print(f"Summarize falcon results took {summarize_time - falcon_end_time:.2f} s.")

        #initialize cluster dic
        falcon_mgf_path = os.path.join(os.getcwd(), "falcon.mgf")
        print("[cluster_one_folder] Performing initial clustering...")
        cluster_dic,singletons = initial_cluster_dic(cluster_info_tsv, falcon_mgf_path, folder)
        cluster_dic_time = time.time()
        timing_log['Initial cluster dic'] = cluster_dic_time - summarize_time
        print(f"Cluster dic initialize took {cluster_dic_time - summarize_time:.2f} s.")

        #write consensus mzML file
        n_written = write_mzml(cluster_dic, output_consensus_path, folder, output_dir, sample_threshold=5)
        consensus_end_time = time.time()
        timing_log['Initial round consensus write'] = consensus_end_time - cluster_dic_time
        print(f"Write consensus mzML file took {consensus_end_time - cluster_dic_time:.2f} s.")

        #write singletons
        singletons_mzml_path = os.path.join(output_dir, "singletons.mzML")
        write_singletons_mzml(singletons, singletons_mzml_path, folder, output_dir)
    else:
        # Incremental processing
        # First Falcon run: new data + consensus
        print("[Incremental] Phase 1: non-singleton clustering...")
        input_files = f"{folder}/*.mzML {consensus_path}"
        run_falcon(input_files, "falcon1", precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps)
        falcon1_end_time = time.time()
        timing_log['Falcon phase 1 clustering'] = falcon1_end_time - start_time
        print(f"Falcon phase 1 clustering took {falcon1_end_time - start_time:.2f} s.")

        #summarize phase1 results
        cluster_info1_tsv = summarize_output(output_dir, os.path.join(tool_dir, 'summarize_results.py'), falcon_csv="falcon1.csv")
        summarize1_time = time.time()
        timing_log['Summarize falcon phase 1 results'] = summarize1_time - falcon1_end_time
        print(f"Summarize falcon phase 1 results took {summarize1_time - falcon1_end_time:.2f} s.")

        #load cluster dic time
        cluster_dic, max_existing_cluster_id = load_cluster_dic_optimized(checkpoint_dir)
        load_cluster_dic_time = time.time()
        timing_log['Load cluster dic'] = load_cluster_dic_time - summarize1_time
        print(f"Load cluster dic took {load_cluster_dic_time - summarize1_time:.2f} s.")

        #Update cluster dic phase 1
        cluster_dic, new_singletons1 = update_cluster_dic(cluster_dic, cluster_info1_tsv, "falcon1.mgf", folder)
        update1_cluster_dic_time = time.time()
        timing_log['Update cluster dic phase 1'] = update1_cluster_dic_time - load_cluster_dic_time
        print(f"Update cluster dic phase1 took {update1_cluster_dic_time - load_cluster_dic_time:.2f} s.")

        #write phase1 singletons
        temp_singletons_path = os.path.join(output_dir, "temp_singletons.mzML")
        print(f"[Debug] temp_singletons_path: {temp_singletons_path}")
        print(f"[Debug] output_dir: {output_dir}")
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # In incremental mode, use checkpoint_dir for reading historical spectra
        write_singletons_mzml(new_singletons1, temp_singletons_path, folder, checkpoint_dir)
        write_singletons1_end_time = time.time()
        timing_log['Write phase 1 singletons'] = write_singletons1_end_time - update1_cluster_dic_time
        print(f"Write phase 1 singletons took {write_singletons1_end_time - update1_cluster_dic_time:.2f} s.")

        # Second Falcon run: existing singletons + new temp_singletons
        singletons_mzml_path = os.path.join(checkpoint_dir, "singletons.mzML")
        run_falcon(f"{singletons_mzml_path} {temp_singletons_path}", "falcon2", precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps)
        falcon2_end_time = time.time()
        timing_log['Falcon phase 2 clustering'] = falcon2_end_time - write_singletons1_end_time
        print(f"Falcon phase 2 took {falcon2_end_time - write_singletons1_end_time:.2f} s.")

        #summarize phase2 results
        cluster_info2_tsv = summarize_output(output_dir, os.path.join(tool_dir, 'summarize_results.py'), falcon_csv="falcon2.csv")
        summarize2_time = time.time()
        timing_log['Summarize falcon phase 2 results'] = summarize2_time - falcon2_end_time
        print(f"Summarize falcon phase 2 results took {summarize2_time - falcon2_end_time:.2f} s.")

        #update cluster dic phase 2
        # Remove path_map/combined_map logic, only pass required arguments
        cluster_dic, new_singletons2 = update_cluster_dic(cluster_dic, cluster_info2_tsv, "falcon2.mgf", folder)
        update2_cluster_dic_time = time.time()
        timing_log['Update cluster dic phase2'] = update2_cluster_dic_time - summarize2_time
        print(f"Update cluster dic phase2 took {update2_cluster_dic_time - summarize2_time:.2f} s.")

        #wirte final consensus results
        # In incremental mode, use checkpoint_dir for reading historical spectra
        n_written = write_mzml(cluster_dic, output_consensus_path, folder, checkpoint_dir, sample_threshold=5)
        consensus_end_time = time.time()
        timing_log['Write consensus mzML file'] = consensus_end_time - update2_cluster_dic_time
        print(f"Write consensus mzML file took {consensus_end_time - update2_cluster_dic_time:.2f} s.")

        #write phase2 singletons
        singletons2_mzml_path = os.path.join(output_dir, "singletons.mzML")
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # In incremental mode, use checkpoint_dir for reading historical spectra
        write_singletons_mzml(new_singletons2, singletons2_mzml_path, folder, checkpoint_dir)
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

    #save updated cluster_dic
    cluster_end_time = time.time()
    
    # Save consensus spectra incrementally
    consensus_parquet_path = os.path.join(output_dir, "consensus.parquet")
    if has_checkpoint:
        # Incremental mode: use max cluster ID from loaded data
        save_consensus_incremental(cluster_dic, consensus_parquet_path, max_existing_cluster_id)
    else:
        # Initial mode: no existing clusters, so max_existing_id = 0
        save_consensus_incremental(cluster_dic, consensus_parquet_path, 0)
    
    # Save spectra incrementally (for both initial and incremental modes)
    if has_checkpoint:
        # Incremental mode: collect scans that need to be saved for next batch
        phase2_singletons = new_singletons2  # Second write_singletons_mzml call
        scans_to_save = collect_scans_for_next_batch(
            cluster_dic, folder, phase2_singletons=phase2_singletons
        )
    else:
        # Initial mode: collect all current batch scans from cluster_dic and singletons
        scans_to_save = collect_scans_for_next_batch(
            cluster_dic, folder, phase2_singletons=singletons
        )
    
    # Save current batch spectra to storage system
    if scans_to_save:
        print(f"[cluster_one_folder] Saving {len(scans_to_save)} current batch spectra to storage...")
        # Group by file for efficient reading
        file_to_scans = defaultdict(list)
        for fp, sc in scans_to_save:
            file_to_scans[fp].append(sc)
        
        # Use joblib for parallel processing
        from joblib import Parallel, delayed
        
        def process_file_spectra(fp, scans):
            """Process all scans from a single file"""
            results = []
            try:
                all_spectra = read_mzml(fp)
                scan_to_spectrum = {int(s['scans']): s for s in all_spectra}
                
                for sc in scans:
                    scan_id = int(sc)
                    if scan_id in scan_to_spectrum:
                        s = scan_to_spectrum[scan_id]
                        results.append({
                            'filename': str(fp),
                            'scan': int(sc),
                            'peaks': s['peaks'],
                            'precursor_mz': s.get('precursor_mz', 0),
                            'rtinseconds': s.get('rtinseconds', 0),
                            'charge': s.get('charge', 0)
                        })
            except Exception as e:
                print(f"[Warning] Failed to process file {fp}: {e}")
            return results
        
        # Parallel processing
        available_cores = os.cpu_count() or 1
        N_JOBS = max(40, min(available_cores // 2, 96))
        
        all_results = Parallel(n_jobs=N_JOBS)(
            delayed(process_file_spectra)(fp, scans) 
            for fp, scans in file_to_scans.items()
        )
        
        # Collect all results
        all_spectra = []
        for file_results in all_results:
            all_spectra.extend(file_results)
        
        # Store using memory-efficient storage with proper incremental update
        if all_spectra:
            # Use output directory directly, not subdirectory
            if has_checkpoint:
                # Incremental mode: copy existing storage from checkpoint to output, then update
                checkpoint_storage_dir = checkpoint_dir  # Checkpoint storage is in checkpoint_dir directly
                
                # Copy existing storage files to output directory
                import shutil
                checkpoint_files = ['spectra.bin', 'spectra_index.db']
                for file_name in checkpoint_files:
                    checkpoint_file = os.path.join(checkpoint_storage_dir, file_name)
                    output_file = os.path.join(output_dir, file_name)
                    if os.path.exists(checkpoint_file):
                        shutil.copy2(checkpoint_file, output_file)
                        print(f"[cluster_one_folder] Copied {file_name} from checkpoint to output")
                
                # Now use output directory for incremental update
                storage = get_spectrum_storage(output_dir)
            else:
                # Initial mode: create new storage in output directory
                storage = get_spectrum_storage(output_dir)
            
            storage.store_spectra_batch(all_spectra)
            stats = storage.get_storage_stats()
            print(f"[cluster_one_folder] Stored {len(all_spectra)} spectra, total: {stats['spectrum_count']}, size: {stats['total_size_mb']:.1f}MB")
    
    # Save scan_list and sample_list to feather files, and spectra to memory-efficient storage
    # Save after all processing is complete (both initial and incremental modes)
    if has_checkpoint:
        # Incremental mode
        save_cluster_dic_optimized(cluster_dic, output_dir, singletons=new_singletons2 if 'new_singletons2' in locals() else singletons, current_batch_folder=folder)
    else:
        # Initial mode
        save_cluster_dic_optimized(cluster_dic, output_dir, singletons=singletons, current_batch_folder=folder)
    
    print(f"[cluster_one_folder] Updated cluster_dic saved at {output_dir}")
    save_cluster_dic_end_time = time.time()
    timing_log['Save cluster dic'] = save_cluster_dic_end_time - cluster_end_time
    print(f"Save cluster dic took {save_cluster_dic_end_time - cluster_end_time:.2f} s.")

    #final results output
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

    timing_log["Total wall time (hours)"] = (end_time - start_time)/ 3600
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

    fieldnames = ['cluster','filename','scan', 'precursor_mz', 'retention_time', 'new_batch']
    with open(out_tsv, 'w', newline='') as f:
        w = DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for cid, cdata in cluster_dic.items():
            for scan_item in cdata['scan_list']:
                if len(scan_item) == 4:  # (filename, scan, precursor_mz, retention_time)
                    fn, sc, precursor_mz, retention_time = scan_item
                else:  # Legacy format (filename, scan)
                    fn, sc = scan_item
                    precursor_mz, retention_time = 0.0, 0.0
                
                is_new = 'yes' if current_batch_files and os.path.basename(fn) in current_batch_files else 'no'
                row = {
                    'cluster': cid,
                    'filename': os.path.basename(fn),
                    'scan': sc,
                    'precursor_mz': precursor_mz,
                    'retention_time': retention_time,
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

    #print the version of the script
    #define the version of the script by "date_version"
    __version__ = f"{datetime.datetime.now().strftime('%Y%m%d')}_1.2.8"
    print(f"Running version: {__version__}")
    print(f"[Debug] Current working directory: {os.getcwd()}")
    
    # Check if this is initial batch (no checkpoint) or incremental batch (has checkpoint)
    checkpoint_storage_bin = os.path.join(args.checkpoint_dir, "spectra.bin")
    has_existing_checkpoint = os.path.exists(checkpoint_storage_bin)
    
    if has_existing_checkpoint:
        # Incremental batch: use existing storage from checkpoint
        storage = get_spectrum_storage(args.checkpoint_dir)
        print(f"[Info] Incremental mode: Using existing storage from checkpoint: {args.checkpoint_dir}")
    else:
        # Initial batch: will create new storage in output_dir during processing
        print(f"[Info] Initial mode: Will create new storage in output directory: {args.output_dir}")
        storage = None  # Will be created in cluster_one_folder
    
    if storage:
        stats = storage.get_storage_stats()
        print(f"[Info] Memory-efficient spectrum storage stats: {stats}")
    else:
        print(f"[Info] No existing storage found, will create new one")
    print(f"[Debug] output_dir: {args.output_dir}")
    print(f"[Debug] folder: {args.folder}")
    print(f"[Debug] checkpoint_dir: {args.checkpoint_dir}")

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
