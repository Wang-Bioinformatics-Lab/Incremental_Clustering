#!/usr/bin/env python3
"""
Hyper-Spec based incremental clustering.

This is a non-invasive alternative to `incremental_clustering_sep_ver.py`:
- Do NOT modify the original script.
- Replace the core clustering engine (Falcon) with Hyper-Spec clustering.
- Hyper-Spec outputs parquet (+ optional representatives mgf). We summarize parquet to the
  `cluster_info.tsv` format that the incremental merge logic expects.

Implementation strategy:
- Reuse the existing incremental merge logic and mzML handling utilities by importing them.
- Only replace:
  - clustering runner (Falcon -> Hyper-Spec)
  - summarizer (summarize_results.py -> summarize_results_hyperspec.py)
  - file naming (falcon*.{csv,mgf} -> hyperspec*.{parquet,mgf})
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from incremental_clustering_sep_ver import (
    initial_cluster_dic,
    update_cluster_dic,
    load_cluster_dic_optimized,
    write_mzml,
    write_singletons_mzml,
    save_consensus_incremental,
    collect_scans_for_next_batch,
    read_mzml,
    read_spectra,
    get_spectrum_storage,
    save_cluster_dic_optimized,
    finalize_results,
)


def _expand_inputs(mzml_pattern: str) -> list[str]:
    toks = mzml_pattern.split()
    out: list[str] = []
    for t in toks:
        if any(ch in t for ch in ["*", "?", "["]):
            out.extend(sorted(glob.glob(t)))
        else:
            out.append(t)
    # keep only existing files
    out = [p for p in out if os.path.exists(p)]
    return out


def _raw_scan_from_native(raw_scan):
    """
    Extract raw scan string from mzML native ID. No parsing—pass through as-is.
    Downstream parses by filename (consensus vs not) and handles underscores.
    """
    s = str(raw_scan).strip()
    if "=" in s:
        s = s.split("=")[-1].strip()
    return s if s else None


def _mzml_to_mgf_one(mzml_path: str, out_mgf_path: str):
    """
    Convert one mzML to MGF for Hyper-Spec. Use raw scan string as SCANS (no encoding).
    TITLE is not used for clustering; we write identifier:index:scan for convention only.
    Downstream parses scan by filename (e.g. consensus "9_1" -> cid via first part).
    """
    if not os.path.exists(mzml_path):
        print(f"[_mzml_to_mgf_one] Warning: File not found, skipping: {mzml_path}")
        return

    try:
        spectra = read_mzml(mzml_path)
    except Exception as e:
        print(f"[_mzml_to_mgf_one] Error reading {mzml_path}: {e}")
        return

    identifier = Path(mzml_path).stem
    with open(out_mgf_path, "w") as f:
        for s in spectra:
            peaks = s.get("peaks", []) or []
            if not peaks:
                continue
            raw_scan = _raw_scan_from_native(s.get("scans"))
            if raw_scan is None:
                print(f"[_mzml_to_mgf_one] Warning: Unparseable scan id '{s.get('scans')}' in {mzml_path}, skipping spectrum")
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


def _make_mgf_dir(input_files: list[str], mgf_dir: str):
    """
    Build MGF directory for Hyper-Spec: copy .mgf as-is, convert .mzML to MGF.
    Returns: (conversion_time_seconds, list of paths that were converted from mzML)
    """
    import time
    import shutil
    os.makedirs(mgf_dir, exist_ok=True)
    mzml_files = [p for p in input_files if p.lower().endswith(".mzml")]
    mgf_files = [p for p in input_files if p.lower().endswith(".mgf")]

    # Copy MGF files as-is (no conversion time)
    for mgf_path in mgf_files:
        if os.path.exists(mgf_path):
            out_mgf = os.path.join(mgf_dir, os.path.basename(mgf_path))
            shutil.copy2(mgf_path, out_mgf)

    # Convert mzML -> MGF in parallel
    convert_time = 0.0
    if mzml_files:
        from joblib import Parallel, delayed
        n_jobs = os.cpu_count() or 1
        def _convert_one(mzml_path: str):
            out_mgf = os.path.join(mgf_dir, f"{Path(mzml_path).stem}.mgf")
            _mzml_to_mgf_one(mzml_path, out_mgf)
        convert_start = time.time()
        Parallel(n_jobs=n_jobs)(delayed(_convert_one)(m) for m in mzml_files)
        convert_time = time.time() - convert_start
    return convert_time


def _copy_representative_mgf_to_output(output_prefix: str, output_dir: str) -> None:
    """Copy representative MGF from CWD to output_dir for debugging."""
    import shutil
    name = f"{output_prefix}_representatives.mgf"
    src = os.path.join(os.getcwd(), name)
    if os.path.isfile(src):
        os.makedirs(output_dir, exist_ok=True)
        dst = os.path.join(output_dir, name)
        shutil.copy2(src, dst)
        print(f"[cluster_one_folder_hyperspec] Copied {name} to {output_dir} for debug")


def run_hyperspec(
    input_files: list[str],
    output_prefix: str,
    eps: float,
    min_mz_range: float,
    min_mz: float,
    max_mz: float,
    use_gpu: bool = True,
    cluster_alg: str = "dbscan",
    cluster_charges: str = "1 2 3",
    cpu_core_preprocess: int = 8,
    cpu_core_cluster: int | None = None,
):
    """
    Run Hyper-Spec clustering. Input can be .mzML (converted to MGF) or .mgf (used as-is).

    Outputs in current working directory:
    - <output_prefix>.parquet
    - <output_prefix>_representatives.mgf (if representative_mgf enabled)

    Returns:
        dict with 'mgf_conversion_time' (seconds; 0 when all inputs are already MGF)
    """
    input_files = [p for p in input_files if os.path.exists(p)]
    if not input_files:
        raise RuntimeError("[run_hyperspec] No input files found (empty list or paths do not exist)")

    mgf_dir = tempfile.mkdtemp(prefix=f"{output_prefix}_mgf_")
    mgf_conversion_time = _make_mgf_dir(input_files, mgf_dir)
    n_mzml = sum(1 for p in input_files if p.lower().endswith(".mzml"))
    n_mgf = len(input_files) - n_mzml
    if n_mzml:
        print(f"[run_hyperspec] Converted {n_mzml} mzML files to MGF in {mgf_conversion_time:.2f}s")
    if n_mgf:
        print(f"[run_hyperspec] Using {n_mgf} MGF files as-is (no conversion)")

    # Detect actual charges present in MGF files to avoid Hyper-Spec crash on empty charge groups
    actual_charges = set()
    for mgf_file in glob.glob(os.path.join(mgf_dir, "*.mgf")):
        with open(mgf_file, "r") as f:
            for line in f:
                if line.startswith("CHARGE="):
                    charge_str = line.split("=")[1].strip().replace("+", "")
                    try:
                        charge_val = int(charge_str)
                        if charge_val > 0:
                            actual_charges.add(charge_val)
                    except ValueError:
                        pass

    # Filter cluster_charges to only include charges that actually exist in the data
    requested_charges = [int(c) for c in cluster_charges.split()]
    available_charges = [str(c) for c in requested_charges if c in actual_charges]
    
    if not available_charges:
        raise RuntimeError(f"[run_hyperspec] No spectra found for requested charges {cluster_charges}. Available charges: {sorted(actual_charges)}")
    
    if set(requested_charges) != actual_charges:
        print(f"[run_hyperspec] Warning: Requested charges {cluster_charges}, but only {available_charges} have spectra. Using {available_charges}")

    # Hyper-Spec is bundled in the docker image at /Hyper-Spec, and its env at /opt/conda/envs/hyper-spec
    hyperspec_python = "/opt/conda/envs/hyper-spec/bin/python"
    hyperspec_main = "/Hyper-Spec/src/main.py"

    out_base = os.path.abspath(output_prefix)
    out_file = out_base  # Hyper-Spec will append .parquet

    cmd = [
        hyperspec_python,
        hyperspec_main,
        mgf_dir,
        out_file,
        f"--cpu_core_preprocess={cpu_core_preprocess}",
        f"--cpu_core_cluster={cpu_core_cluster or os.cpu_count() or 8}",
        f"--cluster_alg={cluster_alg}",
        f"--eps={eps}",
        f"--min_mz_range={min_mz_range}",
        f"--min_mz={min_mz}",
        f"--max_mz={max_mz}",
        "--cluster_charges",
        *available_charges,
        "--representative_mgf",
    ]
    if use_gpu:
        cmd.append("--use_gpu_cluster")

    print(f"[run_hyperspec] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"[run_hyperspec] Hyper-Spec failed with exit code {proc.returncode}")
    
    # Cleanup temporary MGF directory (converted mzML->MGF only; representative MGF is not deleted)
    try:
        shutil.rmtree(mgf_dir)
        print(f"[run_hyperspec] Cleaned up temporary MGF directory: {mgf_dir}")
    except Exception as e:
        print(f"[run_hyperspec] Warning: Failed to cleanup {mgf_dir}: {e}")
    
    return {"mgf_conversion_time": mgf_conversion_time}


def summarize_output_hyperspec(
    output_path: str, tool_dir: str, hyperspec_parquet: str, input_folder: str | None = None
) -> str:
    """
    Summarize <prefix>.parquet into output_summary/cluster_info.tsv (tab-delimited).
    If input_folder is set, filenames in cluster_info use actual extension (.mzML or .mgf).
    """
    output_dir = os.path.join(output_path, "output_summary")
    os.makedirs(output_dir, exist_ok=True)
    summarize_script = os.path.join(tool_dir, "summarize_results_hyperspec.py")
    python_cmd = sys.executable
    cmd = f"{python_cmd} {summarize_script} {hyperspec_parquet} {output_dir}"
    if input_folder and os.path.isdir(input_folder):
        cmd += f" --input_folder {os.path.abspath(input_folder)}"
    print(f"[summarize_output_hyperspec] Running: {cmd}")
    subprocess.check_call(cmd, shell=True)
    return os.path.join(output_dir, "cluster_info.tsv")


def cluster_one_folder_hyperspec(folder, checkpoint_dir, output_dir, tool_dir, precursor_tol, fragment_tol, min_mz_range, min_mz, max_mz, eps):
    """
    Same orchestration as `cluster_one_folder` but with Hyper-Spec replacing Falcon.
    """
    import time
    import datetime
    from collections import defaultdict
    from joblib import Parallel, delayed

    start_time = time.time()
    start_cpu_time = time.process_time()
    timing_log = {}

    consensus_path = os.path.join(checkpoint_dir, "consensus.mzML")
    output_consensus_path = os.path.join(output_dir, "consensus.mzML")

    scan_feather = os.path.join(checkpoint_dir, "scan_list.feather")
    storage_bin = os.path.join(checkpoint_dir, "spectra.bin")
    has_checkpoint = os.path.exists(scan_feather) and os.path.exists(storage_bin)

    os.makedirs(output_dir, exist_ok=True)

    # Discover input spectrum files: support both .mzML and .mgf (exclude consensus.mzML)
    def _batch_spectrum_files(folder_path):
        if not os.path.isdir(folder_path):
            return []
        return sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if (f.endswith(".mzML") or f.endswith(".mgf")) and f != "consensus.mzML"]
        )

    batch_files = _batch_spectrum_files(folder)
    if not batch_files:
        raise RuntimeError(f"[cluster_one_folder_hyperspec] No .mzML or .mgf files in folder: {folder}")

    if not has_checkpoint:
        print("[Initial] Running Hyper-Spec on all new spectra...")
        hs_result = run_hyperspec(batch_files, "hyperspec", eps, min_mz_range, min_mz, max_mz)
        hs_end = time.time()
        timing_log["mzML to MGF conversion (initial)"] = hs_result.get("mgf_conversion_time", 0)
        timing_log["Hyper-Spec initial"] = hs_end - start_time
        _copy_representative_mgf_to_output("hyperspec", output_dir)

        hyperspec_parquet = os.path.join(os.getcwd(), "hyperspec.parquet")
        cluster_info_tsv = summarize_output_hyperspec(output_dir, tool_dir, hyperspec_parquet, input_folder=folder)
        summarize_time = time.time()
        timing_log["summarize initial results"] = summarize_time - hs_end

        reps_mgf_path = os.path.join(os.getcwd(), "hyperspec_representatives.mgf")
        cluster_dic, singletons = initial_cluster_dic(cluster_info_tsv, reps_mgf_path, folder)
        cluster_dic_time = time.time()
        timing_log["Initial cluster dic"] = cluster_dic_time - summarize_time

        write_mzml(cluster_dic, output_consensus_path, folder, output_dir, sample_threshold=5)
        consensus_end = time.time()
        timing_log["Initial round consensus write"] = consensus_end - cluster_dic_time

        singletons_mzml_path = os.path.join(output_dir, "singletons.mzML")
        write_singletons_mzml(singletons, singletons_mzml_path, folder, output_dir)
    else:
        # Phase 1: new batch files + consensus
        print("[Incremental] Phase 1: non-singleton clustering (Hyper-Spec)...")
        phase1_inputs = batch_files + ([consensus_path] if os.path.isfile(consensus_path) else [])
        hs1_result = run_hyperspec(phase1_inputs, "hyperspec1", eps, min_mz_range, min_mz, max_mz)
        hs1_end = time.time()
        timing_log["mzML to MGF conversion (phase 1)"] = hs1_result.get("mgf_conversion_time", 0)
        timing_log["Hyper-Spec phase 1 clustering"] = hs1_end - start_time
        _copy_representative_mgf_to_output("hyperspec1", output_dir)

        hyperspec1_parquet = os.path.join(os.getcwd(), "hyperspec1.parquet")
        cluster_info1_tsv = summarize_output_hyperspec(output_dir, tool_dir, hyperspec1_parquet, input_folder=folder)
        summarize1_time = time.time()
        timing_log["Summarize Hyper-Spec phase 1 results"] = summarize1_time - hs1_end

        cluster_dic, max_existing_cluster_id = load_cluster_dic_optimized(checkpoint_dir)
        load_time = time.time()
        timing_log["Load cluster dic"] = load_time - summarize1_time

        cluster_dic, new_singletons1 = update_cluster_dic(cluster_dic, cluster_info1_tsv, "hyperspec1_representatives.mgf", folder)
        update1_time = time.time()
        timing_log["Update cluster dic phase 1"] = update1_time - load_time
        # If there are no singletons from previous checkpoint or current batch, skip phase 2 entirely
        singletons_mzml_path = os.path.join(checkpoint_dir, "singletons.mzML")
        has_prev_singletons = os.path.exists(singletons_mzml_path)
        has_new_singletons1 = bool(new_singletons1)

        if not has_prev_singletons and not has_new_singletons1:
            print("[cluster_one_folder_hyperspec] No singletons from previous checkpoint or current batch; skipping phase 2 clustering.")
            # Directly write consensus and checkpoint without any singletons handling
            write_mzml(cluster_dic, output_consensus_path, folder, checkpoint_dir, sample_threshold=5)
            consensus_end = time.time()
            timing_log["Write consensus mzML file (no singletons)"] = consensus_end - update1_time
        else:
            # Phase 2: existing singletons + new temp_singletons
            temp_singletons_path = os.path.join(output_dir, "temp_singletons.mzML")
            write_singletons_mzml(new_singletons1, temp_singletons_path, folder, checkpoint_dir)
            write_singletons1_end = time.time()
            timing_log["Write phase 1 singletons"] = write_singletons1_end - update1_time

            # Only include singletons.mzML if it exists (may not exist if previous rounds had no singletons)
            if has_prev_singletons:
                phase2_inputs = [singletons_mzml_path, temp_singletons_path]
            else:
                phase2_inputs = [temp_singletons_path]
                print(f"[cluster_one_folder_hyperspec] No existing singletons.mzML found, using only temp_singletons")

            hs2_result = run_hyperspec(phase2_inputs, "hyperspec2", eps, min_mz_range, min_mz, max_mz)
            hs2_end = time.time()
            timing_log["mzML to MGF conversion (phase 2)"] = hs2_result.get("mgf_conversion_time", 0)
            timing_log["Hyper-Spec phase 2 clustering"] = hs2_end - write_singletons1_end
            _copy_representative_mgf_to_output("hyperspec2", output_dir)

            hyperspec2_parquet = os.path.join(os.getcwd(), "hyperspec2.parquet")
            cluster_info2_tsv = summarize_output_hyperspec(output_dir, tool_dir, hyperspec2_parquet, input_folder=folder)
            summarize2_time = time.time()
            timing_log["Summarize Hyper-Spec phase 2 results"] = summarize2_time - hs2_end

            cluster_dic, new_singletons2 = update_cluster_dic(cluster_dic, cluster_info2_tsv, "hyperspec2_representatives.mgf", folder)
            update2_time = time.time()
            timing_log["Update cluster dic phase 2"] = update2_time - summarize2_time

            # Save consensus to output_dir (which becomes checkpoint for next batch)
            write_mzml(cluster_dic, output_consensus_path, folder, checkpoint_dir, sample_threshold=5)
            consensus_end = time.time()
            timing_log["Write consensus mzML file"] = consensus_end - update2_time

            # Save singletons to output_dir
            singletons2_mzml_path = os.path.join(output_dir, "singletons.mzML")
            write_singletons_mzml(new_singletons2, singletons2_mzml_path, folder, checkpoint_dir)
            write_singletons2_end = time.time()
            timing_log["Write phase 2 singletons"] = write_singletons2_end - consensus_end

    # Save consensus spectra incrementally
    # Always save to output_dir (which becomes the checkpoint for next batch)
    consensus_parquet_path = os.path.join(output_dir, "consensus.parquet")
    if has_checkpoint:
        save_consensus_incremental(cluster_dic, consensus_parquet_path, max_existing_cluster_id)
    else:
        save_consensus_incremental(cluster_dic, consensus_parquet_path, 0)

    # Save spectra for next batch (same logic as original)
    if has_checkpoint:
        # If phase 2 was skipped, fall back to phase 1 singletons (or empty list)
        if "new_singletons2" in locals():
            phase2_singletons = new_singletons2
        else:
            phase2_singletons = new_singletons1
        scans_to_save = collect_scans_for_next_batch(cluster_dic, folder, phase2_singletons=phase2_singletons)
    else:
        scans_to_save = collect_scans_for_next_batch(cluster_dic, folder, phase2_singletons=singletons)

    if scans_to_save:
        print(f"[cluster_one_folder_hyperspec] Saving {len(scans_to_save)} current batch spectra to storage...")
        file_to_scans = defaultdict(list)
        for fp, sc in scans_to_save:
            file_to_scans[fp].append(sc)

        def process_file_spectra(fp, scans):
            results = []
            try:
                all_spectra = read_spectra(fp)
                scan_to_spectrum = {int(s["scans"]): s for s in all_spectra}
                for sc in scans:
                    scan_id = int(sc)
                    if scan_id in scan_to_spectrum:
                        s = scan_to_spectrum[scan_id]
                        results.append(
                            {
                                "filename": str(fp),
                                "scan": int(sc),
                                "peaks": s["peaks"],
                                "precursor_mz": s.get("precursor_mz", 0),
                                "rtinseconds": s.get("rtinseconds", 0),
                                "charge": s.get("charge", 0),
                            }
                        )
            except Exception as e:
                print(f"[Warning] Failed to process file {fp}: {e}")
            return results

        available_cores = os.cpu_count() or 1
        n_jobs = max(40, min(available_cores // 2, 96))
        all_results = Parallel(n_jobs=n_jobs)(delayed(process_file_spectra)(fp, scans) for fp, scans in file_to_scans.items())
        all_spectra = []
        for fr in all_results:
            all_spectra.extend(fr)

        if all_spectra:
            # Always save to output_dir (which becomes checkpoint for next batch)
            storage = get_spectrum_storage(output_dir)
            storage.store_spectra_batch(all_spectra)

    # Save cluster dic summary (same as original)
    # Always save to output_dir (which becomes checkpoint for next batch)
    if has_checkpoint:
        save_cluster_dic_optimized(
            cluster_dic,
            output_dir,
            singletons=new_singletons2 if "new_singletons2" in locals() else singletons,
            current_batch_folder=folder,
        )
    else:
        save_cluster_dic_optimized(cluster_dic, output_dir, singletons=singletons, current_batch_folder=folder)

    current_batch_files = {
        f for f in os.listdir(folder)
        if (f.endswith(".mzML") or f.endswith(".mgf")) and f != "consensus.mzML"
    }
    finalize_results(cluster_dic, output_dir, current_batch_files)

    end_time = time.time()
    end_cpu_time = time.process_time()
    timing_log["Total wall time (hours)"] = (end_time - start_time) / 3600
    timing_log["Total CPU time (hours)"] = (end_cpu_time - start_cpu_time) / 3600

    timing_file = os.path.join(output_dir, "timing_report.txt")
    with open(timing_file, "w") as f:
        f.write(f"Run Timestamp: {datetime.datetime.now()}\n")
        for k, v in timing_log.items():
            f.write(f"{k}: {v:.2f}\n")
    print(f"[Timing] Full timing report written to: {timing_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to a folder containing *.mzML")
    parser.add_argument("--checkpoint_dir", default="./checkpoint", help="Checkpoint directory")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--tool_dir", default="./bin", help="Tool scripts directory")

    # Keep Falcon-like params for compatibility, but only pass through what Hyper-Spec uses.
    parser.add_argument("--precursor_tol", default="20 ppm", help="Unused (kept for compatibility)")
    parser.add_argument("--fragment_tol", type=float, default=0.05, help="Unused (kept for compatibility)")
    parser.add_argument("--min_mz_range", type=float, default=0, help="Hyper-Spec min_mz_range")
    parser.add_argument("--min_mz", type=float, default=0, help="Hyper-Spec min_mz")
    parser.add_argument("--max_mz", type=float, default=4000, help="Hyper-Spec max_mz")
    parser.add_argument("--eps", type=float, default=0.3, help="Hyper-Spec eps")
    args = parser.parse_args()

    print(f"[incremental_clustering_hyper_spec] CWD: {os.getcwd()}")
    print(f"[incremental_clustering_hyper_spec] folder={args.folder} checkpoint_dir={args.checkpoint_dir} output_dir={args.output_dir}")

    # Prime storage decision like original script
    checkpoint_storage_bin = os.path.join(args.checkpoint_dir, "spectra.bin")
    if os.path.exists(checkpoint_storage_bin):
        _ = get_spectrum_storage(args.checkpoint_dir)
        print(f"[Info] Incremental mode: Using existing storage from checkpoint: {args.checkpoint_dir}")
    else:
        print(f"[Info] Initial mode: Will create new storage in output directory: {args.output_dir}")

    cluster_one_folder_hyperspec(
        args.folder,
        args.checkpoint_dir,
        args.output_dir,
        args.tool_dir,
        args.precursor_tol,
        args.fragment_tol,
        args.min_mz_range,
        args.min_mz,
        args.max_mz,
        args.eps,
    )


if __name__ == "__main__":
    main()

