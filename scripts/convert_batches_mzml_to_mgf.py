#!/usr/bin/env python3
"""
Convert mzML batches to MGF using the same logic as the workflow (_mzml_to_mgf_one).
Uses only pymzml so it can run without the full conda env (no pyopenms).
Usage:
  python convert_batches_mzml_to_mgf.py <input_batches_dir> <output_mgf_dir> [batch_numbers...]
  e.g. python convert_batches_mzml_to_mgf.py /path/to/Test_large_batch_20batches /path/to/Test_large_batch_20batches_mgf 1 2 3 4 5
  If batch_numbers omitted, converts batches 1-5.
"""
import os
import sys
from pathlib import Path

try:
    import pymzml
except ImportError:
    print("Error: pymzml is required. Install with: pip install pymzml")
    sys.exit(1)


def _raw_scan_from_native(raw_scan):
    s = str(raw_scan).strip()
    if "=" in s:
        s = s.split("=")[-1].strip()
    return s if s else None


def read_mzml_spectra(filepath):
    """Read mzML and return list of spectrum dicts (same shape as workflow's read_mzml)."""
    spectra = []
    run = pymzml.run.Reader(filepath, build_index_from_scratch=True)
    for spectrum in run:
        if spectrum.get("ms level") != 2:
            continue
        try:
            prec = spectrum.selected_precursors[0] if spectrum.selected_precursors else {}
            rt = spectrum.scan_time[0] if spectrum.scan_time else 0
            peaks = list(spectrum.peaks("centroided")) if hasattr(spectrum, "peaks") else []
            spectra.append({
                "peaks": peaks,
                "precursor_mz": prec.get("mz", 0.0),
                "rtinseconds": rt,
                "scans": spectrum.get("id"),
                "charge": prec.get("charge", 0),
            })
        except Exception:
            continue
    return spectra


def mzml_to_mgf_one(mzml_path: str, out_mgf_path: str):
    """Same logic as workflow _mzml_to_mgf_one."""
    if not os.path.exists(mzml_path):
        return
    try:
        spectra = read_mzml_spectra(mzml_path)
    except Exception as e:
        print(f"[Warn] Error reading {mzml_path}: {e}")
        return
    identifier = Path(mzml_path).stem
    with open(out_mgf_path, "w") as f:
        for s in spectra:
            peaks = s.get("peaks", []) or []
            if not peaks:
                continue
            raw_scan = _raw_scan_from_native(s.get("scans"))
            if raw_scan is None:
                continue
            pepmass = float(s.get("precursor_mz", 0.0))
            rt = float(s.get("rtinseconds", 0.0))
            charge = s.get("charge", 0) or 0
            try:
                charge = int(str(charge).replace("+", ""))
            except Exception:
                charge = 0
            f.write("BEGIN IONS\n")
            f.write(f"TITLE={identifier}:index:{raw_scan}\n")
            f.write(f"SCANS={raw_scan}\n")
            f.write(f"PEPMASS={pepmass}\n")
            f.write(f"RTINSECONDS={rt}\n")
            f.write(f"CHARGE={charge}+\n")
            for mz, inten in peaks:
                f.write(f"{mz} {inten}\n")
            f.write("END IONS\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_batches_mzml_to_mgf.py <input_batches_dir> <output_mgf_dir> [batch_numbers...]")
        print("Example: python convert_batches_mzml_to_mgf.py /data/.../Test_large_batch_20batches /data/.../Test_large_batch_20batches_mgf 1 2 3 4 5")
        sys.exit(1)
    input_base = Path(sys.argv[1])
    output_base = Path(sys.argv[2])
    if len(sys.argv) > 3:
        batch_nums = [int(x) for x in sys.argv[3:]]
    else:
        batch_nums = [1, 2, 3, 4, 5]
    os.makedirs(output_base, exist_ok=True)
    total_files = 0
    for b in batch_nums:
        batch_name = f"batch_{b}"
        src_dir = input_base / batch_name
        dst_dir = output_base / batch_name
        if not src_dir.is_dir():
            print(f"[Skip] {src_dir} not found")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        mzml_files = sorted(src_dir.glob("*.mzML"))
        try:
            from joblib import Parallel, delayed
            n_jobs = min(32, max(1, (os.cpu_count() or 4) - 2))
            Parallel(n_jobs=n_jobs)(
                delayed(mzml_to_mgf_one)(str(p), str(dst_dir / (p.stem + ".mgf")))
                for p in mzml_files
            )
        except Exception:
            for mzml_path in mzml_files:
                out_mgf = dst_dir / (mzml_path.stem + ".mgf")
                mzml_to_mgf_one(str(mzml_path), str(out_mgf))
        total_files += len(mzml_files)
        print(f"[Done] {batch_name}: {len(mzml_files)} files -> {dst_dir}")
    print(f"[Total] Converted {total_files} mzML files to MGF under {output_base}")


if __name__ == "__main__":
    main()
