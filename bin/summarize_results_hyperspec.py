#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd


def _identifier_to_filename(identifier: str, stem_to_basename: dict | None = None) -> str:
    """
    Map Hyper-Spec 'identifier' to filename used by incremental workflow.
    - For consensus: always consensus.mzML.
    - If stem_to_basename is provided (from --input_folder), use actual extension (.mzML or .mgf).
    - Otherwise default to identifier.mzML for backward compatibility.
    """
    if identifier == "consensus":
        return "consensus.mzML"
    if stem_to_basename is not None and identifier in stem_to_basename:
        return stem_to_basename[identifier]
    return f"{identifier}.mzML"


def main():
    parser = argparse.ArgumentParser(description="Summarize Hyper-Spec parquet results into cluster_info.tsv")
    parser.add_argument("hyperspec_parquet", help="Hyper-Spec output parquet (e.g. hyperspec.parquet)")
    parser.add_argument("output_summary_folder", help="Output folder (will write cluster_info.tsv)")
    parser.add_argument(
        "--input_folder",
        default=None,
        help="Folder containing input spectrum files (.mzML or .mgf). If set, filename column uses actual extension.",
    )
    args = parser.parse_args()

    in_path = Path(args.hyperspec_parquet)
    out_dir = Path(args.output_summary_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_to_basename = None
    if args.input_folder and os.path.isdir(args.input_folder):
        stem_to_basename = {}
        for f in os.listdir(args.input_folder):
            if f.endswith(".mzML") or f.endswith(".mgf"):
                stem = os.path.splitext(f)[0]
                stem_to_basename[stem] = f
        if stem_to_basename:
            print(f"[summarize_results_hyperspec] Using {len(stem_to_basename)} input filenames from {args.input_folder}")

    df = pd.read_parquet(in_path)

    required = {
        "identifier",
        "scan",
        "precursor_mz",
        "retention_time",
        "precursor_charge",
        "cluster",
        "is_representative",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Hyper-Spec parquet: {sorted(missing)}")

    df = df.copy()
    df["identifier"] = df["identifier"].astype(str)
    df["filename"] = df["identifier"].map(lambda x: _identifier_to_filename(x, stem_to_basename))

    # Mark singletons (cluster size = 1) as cluster = -1 to match Falcon behavior
    cluster_sizes = df.groupby("cluster").size()
    singleton_clusters = cluster_sizes[cluster_sizes == 1].index
    df.loc[df["cluster"].isin(singleton_clusters), "cluster"] = -1
    
    print(f"[summarize_results_hyperspec] Found {len(singleton_clusters)} singleton clusters (marked as -1)")
    print(f"[summarize_results_hyperspec] Total clusters: {df['cluster'].nunique()}, Singletons: {(df['cluster'] == -1).sum()}")

    # Keep the columns that incremental_clustering expects (and extra columns for debugging)
    out_cols = [
        "cluster",
        "filename",
        "scan",
        "precursor_mz",
        "retention_time",
        "precursor_charge",
        "is_representative",
        "identifier",
    ]

    out_path = out_dir / "cluster_info.tsv"
    df.loc[:, out_cols].to_csv(out_path, sep="\t", index=False)
    print(f"[summarize_results_hyperspec] Wrote: {out_path}")


if __name__ == "__main__":
    main()

