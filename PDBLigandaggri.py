# scripts/build_pdb_ligand_table.py
"""
Build a two-column table (PDB_ID, ligand_resname) from a folder of .txt files.

Each .txt file name (without extension) is treated as the ligand code.
Each file's contents are comma-separated (and/or newline/space-separated) PDB IDs.

Example:
  BTN.txt -> "1AVD, 1BIB, 1DF8"
  Yields rows: (1AVD, BTN), (1BIB, BTN), (1DF8, BTN)

Usage:
  python scripts/build_pdb_ligand_table.py --input-dir /path/to/folder --output output.xlsx
  python scripts/build_pdb_ligand_table.py -i . -o pairs.csv

Why certain choices:
- Regex-based split is resilient to mixed separators.
- PDB ID validation avoids accidental garbage rows.
- Uppercasing normalizes IDs/ligand codes across sources.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pandas is required. Install with `pip install pandas openpyxl`."
    ) from exc


PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")  # strict 4-char PDB ID
SPLIT_RE = re.compile(r"[,\s]+")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate (PDB_ID, ligand_resname) table from ligand->PDB mapping text files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .txt files. Each file name is a ligand code; file content lists PDB IDs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file path (.xlsx for Excel or .csv).",
    )
    parser.add_argument(
        "-e",
        "--ext",
        type=str,
        default=".txt",
        help="File extension to scan (default: .txt).",
    )
    parser.add_argument(
        "--allow-nonpdb",
        action="store_true",
        help="Allow tokens that don't look like 4-char PDB IDs (disables validation).",
    )
    return parser.parse_args(list(argv))


def tokenize_ids(text: str, validate: bool = True) -> List[str]:
    tokens = [t.strip() for t in SPLIT_RE.split(text) if t.strip()]
    tokens = [t.upper() for t in tokens]
    if validate:
        tokens = [t for t in tokens if PDB_ID_RE.match(t)]
    return tokens


def rows_from_file(file_path: Path, validate: bool = True) -> List[Tuple[str, str]]:
    ligand = file_path.stem.upper()
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for odd encodings
        text = file_path.read_text(encoding="latin-1")
    pdb_ids = tokenize_ids(text, validate=validate)
    return [(pid, ligand) for pid in pdb_ids]


def discover_files(input_dir: Path, ext: str) -> List[Path]:
    if not ext.startswith("."):
        ext = f".{ext}"
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()])


def build_dataframe(files: List[Path], allow_nonpdb: bool) -> "pd.DataFrame":
    all_rows: List[Tuple[str, str]] = []
    for f in files:
        rows = rows_from_file(f, validate=not allow_nonpdb)
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(columns=["PDB_ID", "ligand_resname"])

    df = pd.DataFrame(all_rows, columns=["PDB_ID", "ligand_resname"])
    df = df.drop_duplicates().sort_values(["PDB_ID", "ligand_resname"]).reset_index(drop=True)
    return df


def write_output(df: "pd.DataFrame", output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        # Avoid style issues to keep file light/portable
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)
    else:
        df.to_csv(output_path, index=False)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Error: Input dir not found or not a directory: {args.input_dir}", file=sys.stderr)
        return 2

    files = discover_files(args.input_dir, args.ext)
    if not files:
        print(f"No files with extension {args.ext} found in {args.input_dir}", file=sys.stderr)
        return 1

    df = build_dataframe(files, allow_nonpdb=args.allow_nonpdb)
    write_output(df, args.output)

    print(
        f"Wrote {len(df):,} rows from {len(files)} file(s) to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
