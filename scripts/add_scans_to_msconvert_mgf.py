#!/usr/bin/env python3
"""
Add SCANS= field to MGF files produced by msconvert (no SCANS in output).

Msconvert TITLE format is e.g. "EK_Q_07.4.4.1" (no "scan="). We use 1-based
spectrum index as SCANS so each spectrum has a unique integer. Writes
SCANS= immediately after TITLE; skips blocks that already have SCANS=.

Usage:
  python add_scans_to_msconvert_mgf.py <input.mgf> [output.mgf]
  If output omitted, overwrites input. Use a different output path to keep
  the original msconvert MGF unchanged.
"""

import sys
from pathlib import Path


def add_scans_to_mgf(mgf_path: str, out_path: str | None = None) -> tuple[int, int]:
    """
    Add SCANS=<1-based index> after each TITLE in MGF. Return (added, skipped).
    """
    mgf_path = Path(mgf_path)
    out_path = Path(out_path) if out_path else mgf_path
    lines = mgf_path.read_text(encoding="utf-8").splitlines(keepends=True)

    output: list[str] = []
    i = 0
    added = 0
    skipped = 0
    spec_index = 0

    while i < len(lines):
        line = lines[i]
        output.append(line)

        if line.strip() == "BEGIN IONS":
            spec_index += 1
            # Look at next line: expect TITLE=
            j = i + 1
            has_scans = False
            title_idx = -1
            while j < len(lines):
                l = lines[j]
                if l.startswith("SCANS="):
                    has_scans = True
                    break
                if l.startswith("TITLE="):
                    title_idx = len(output) + (j - i - 1)
                if l.strip() == "END IONS" or l.startswith(("PEPMASS=", "CHARGE=")) and not l.startswith("TITLE="):
                    break
                j += 1

            if not has_scans and title_idx >= 0:
                # Find position: add SCANS right after TITLE line
                k = i + 1
                while k < len(lines) and not lines[k].startswith("TITLE="):
                    output.append(lines[k])
                    k += 1
                if k < len(lines):
                    output.append(lines[k])  # TITLE line
                    output.append(f"SCANS={spec_index}\n")
                    added += 1
                    k += 1
                # Consume rest of block until we've added all lines
                while k < len(lines) and lines[k].strip() != "END IONS":
                    output.append(lines[k])
                    k += 1
                if k < len(lines):
                    output.append(lines[k])
                    k += 1
                i = k
                continue
            else:
                if has_scans:
                    skipped += 1

        i += 1

    out_path.write_text("".join(output), encoding="utf-8")
    return added, skipped


def _add_scans_simple(mgf_path: Path, out_path: Path) -> tuple[int, int]:
    """Simpler: scan line-by-line, on TITLE= append SCANS= if block has no SCANS."""
    text = mgf_path.read_text(encoding="utf-8")
    out_lines: list[str] = []
    added = 0
    skipped = 0
    spec_index = 0
    i = 0
    line_list = text.splitlines(keepends=True)

    while i < len(line_list):
        line = line_list[i]
        if line.strip() == "BEGIN IONS":
            spec_index += 1
            out_lines.append(line)
            i += 1
            block_lines: list[str] = []
            while i < len(line_list) and line_list[i].strip() != "END IONS":
                block_lines.append(line_list[i])
                i += 1
            if i < len(line_list):
                block_lines.append(line_list[i])
                i += 1

            has_scans = any(l.startswith("SCANS=") for l in block_lines)
            if has_scans:
                skipped += 1
                out_lines.extend(block_lines)
                continue

            # Insert SCANS= after TITLE=
            wrote = False
            for l in block_lines:
                out_lines.append(l)
                if not wrote and l.startswith("TITLE="):
                    out_lines.append(f"SCANS={spec_index}\n")
                    added += 1
                    wrote = True
            continue

        out_lines.append(line)
        i += 1

    out_path.write_text("".join(out_lines), encoding="utf-8")
    return added, skipped


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_scans_to_msconvert_mgf.py <input.mgf> [output.mgf]")
        print("  If output omitted, overwrites input.")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    a, s = _add_scans_simple(Path(inp), Path(out) if out else Path(inp))
    print(f"Added SCANS= for {a} spectra, skipped {s} (already had SCANS).")
