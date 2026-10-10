"""Extract per-region range features from NCANDA-style fMRI time-course text files.

Expected demographic CSV columns, in order:
participant_id, sex, age, scanner, heavy_drinker
The first row is treated as a header. Confirm these columns and identifiers against
the authorized dataset documentation before use.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def load_demographics(path: Path) -> pd.DataFrame:
    demo = pd.read_csv(path, dtype={"participant_id": str})
    expected = {"participant_id", "sex", "age", "scanner", "heavy_drinker"}
    missing = expected - set(demo.columns)
    if missing:
        raise ValueError(
            f"Demographic CSV is missing columns {sorted(missing)}. "
            "Rename/map columns explicitly after checking the data dictionary."
        )
    demo["participant_id"] = demo["participant_id"].astype(str).str.strip()
    if demo["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant_id values found in demographics.")
    return demo


def participant_id_from_path(path: Path) -> str:
    # Original scripts use the filename stem as the demographic key.
    return path.stem


def extract_one(path: Path, time_length: int | None = 269) -> dict[str, float]:
    matrix = np.loadtxt(path, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D time-by-region numeric matrix.")
    if time_length is not None:
        if matrix.shape[0] < time_length:
            raise ValueError(
                f"{path} has {matrix.shape[0]} time points; expected at least {time_length}."
            )
        matrix = matrix[:time_length, :]
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{path} contains an empty matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} contains NaN or infinite values.")
    output = {}
    # Range (max - min) follows the feature extraction described in the research code.
    for region in range(matrix.shape[1]):
        output[f"region_{region + 1}_range"] = float(
            np.max(matrix[:, region]) - np.min(matrix[:, region])
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-course-dir", required=True, type=Path)
    parser.add_argument("--demographics", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/features.csv"), type=Path)
    parser.add_argument("--time-length", type=int, default=269)
    parser.add_argument("--include-demographics", action="store_true",
                        help="Include age/sex/scanner as predictors; default excludes them.")
    args = parser.parse_args()

    if not args.time_course_dir.is_dir():
        raise FileNotFoundError(f"Time-course directory not found: {args.time_course_dir}")
    demos = load_demographics(args.demographics)
    demo_by_id = demos.set_index("participant_id")
    paths = sorted(args.time_course_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {args.time_course_dir}")

    rows, skipped = [], []
    for path in paths:
        pid = participant_id_from_path(path)
        if pid not in demo_by_id.index:
            skipped.append((pid, "no matching demographic row"))
            continue
        try:
            row = {"participant_id": pid, **extract_one(path, args.time_length)}
        except (OSError, ValueError) as exc:
            skipped.append((pid, str(exc)))
            continue
        demo = demo_by_id.loc[pid]
        label = str(demo["heavy_drinker"]).strip().lower()
        if label in {"y", "yes", "1", "true", "heavy", "heavy drinker"}:
            row["label"] = 1
        elif label in {"n", "no", "0", "false", "non-heavy", "non-heavy drinker"}:
            row["label"] = 0
        else:
            skipped.append((pid, f"unrecognized heavy_drinker label: {demo['heavy_drinker']!r}"))
            continue

        if args.include_demographics:
            row["age"] = pd.to_numeric(demo["age"], errors="coerce")
            row["sex_male"] = 1 if str(demo["sex"]).strip().upper() == "M" else 0
            scanner = str(demo["scanner"]).strip().lower()
            row["scanner_ge"] = 1 if scanner == "ge" else 0
        rows.append(row)

    if not rows:
        raise RuntimeError("No usable participants matched. Check file naming and demographic IDs.")
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} participants to {args.output}")
    print(f"Class counts:\n{result['label'].value_counts().sort_index().to_string()}")
    if skipped:
        skip_path = args.output.with_name(args.output.stem + "_skipped.csv")
        pd.DataFrame(skipped, columns=["participant_id", "reason"]).to_csv(skip_path, index=False)
        print(f"Skipped {len(skipped)} files; reasons saved to {skip_path}")
    print("Verify demographic schema, participant matching, and inclusion rules before analysis.")


if __name__ == "__main__":
    main()
