#!/usr/bin/env python3
"""Convert ExtraSensory .dat recordings to CSV and resample them.

Each source file must contain four whitespace-separated numeric columns:
sensor timestamp, X, Y, and Z.  The filename prefix is treated as the Unix
recording timestamp (for example, ``1440980102.m_raw_acc.dat``).

The program is modality agnostic: set ``--modality raw_acc`` for the current
accelerometer files and use the gyro filename token when gyro files arrive.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FILENAME_PATTERN = re.compile(r"^(?P<timestamp>\d+)\.m_(?P<modality>.+)\.dat$")


@dataclass(frozen=True)
class Recording:
    source: Path
    user_id: str
    recording_timestamp: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--modality",
        required=True,
        help="Filename token after m_, e.g. raw_acc or proc_gyro.",
    )
    parser.add_argument("--source-rate-hz", type=float, default=40.0)
    parser.add_argument("--target-rate-hz", type=float, default=25.0)
    parser.add_argument(
        "--expected-source-samples",
        type=int,
        default=800,
        help="Warn when a capture does not have this many rows (default: 800).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output CSV files that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many files (useful for a quick validation run).",
    )
    return parser.parse_args()


def find_recordings(input_dir: Path, modality: str) -> list[Recording]:
    suffix = f".m_{modality}.dat"
    recordings: list[Recording] = []
    for source in sorted(input_dir.rglob(f"*{suffix}")):
        match = FILENAME_PATTERN.match(source.name)
        if match is None:
            continue
        recordings.append(
            Recording(
                source=source,
                user_id=source.parent.name,
                recording_timestamp=int(match.group("timestamp")),
            )
        )
    return recordings


def load_dat(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path}: cannot parse whitespace-separated numeric data") from exc
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{path}: expected exactly 4 columns (timestamp, x, y, z); got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains non-finite values")
    return values


def resample_rows(values: np.ndarray, output_samples: int) -> np.ndarray:
    """Linearly resample an N-by-4 recording to exactly output_samples rows.

    The source sensor clock is preserved as a signal in column 0.  Axis values
    are resampled by sample position, rather than by the sometimes irregular
    device clock, because the challenge defines every capture as 800 samples
    acquired at the nominal source rate.
    """
    input_positions = np.arange(len(values), dtype=np.float64)
    output_positions = np.linspace(0, len(values) - 1, output_samples)
    return np.column_stack(
        [np.interp(output_positions, input_positions, values[:, column]) for column in range(4)]
    )


def write_csv(path: Path, values: np.ndarray, recording_timestamp: int, rate_hz: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "recording_timestamp_s",
                "elapsed_s",
                "sensor_timestamp",
                "x",
                "y",
                "z",
            ]
        )
        for index, row in enumerate(values):
            writer.writerow(
                [
                    recording_timestamp,
                    f"{index / rate_hz:.6f}",
                    f"{row[0]:.9f}",
                    f"{row[1]:.9f}",
                    f"{row[2]:.9f}",
                    f"{row[3]:.9f}",
                ]
            )


def output_paths(output_dir: Path, recording: Recording, modality: str) -> tuple[Path, Path]:
    stem = recording.source.stem
    native = output_dir / f"{modality}_csv" / recording.user_id / f"{stem}.csv"
    resampled = output_dir / f"{modality}_25hz" / recording.user_id / f"{stem}.csv"
    return native, resampled


def main() -> int:
    args = parse_args()
    if args.source_rate_hz <= 0 or args.target_rate_hz <= 0:
        raise ValueError("Sample rates must be positive.")
    target_samples = round(args.expected_source_samples * args.target_rate_hz / args.source_rate_hz)
    recordings = find_recordings(args.input_dir, args.modality)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        recordings = recordings[: args.limit]
    if not recordings:
        print(f"No .m_{args.modality}.dat files found under {args.input_dir}", file=sys.stderr)
        return 1

    manifest_rows: list[dict[str, str | int | float]] = []
    warnings = 0
    for index, recording in enumerate(recordings, start=1):
        values = load_dat(recording.source)
        native_path, resampled_path = output_paths(args.output_dir, recording, args.modality)
        if not args.overwrite and (native_path.exists() or resampled_path.exists()):
            raise FileExistsError(f"Output exists for {recording.source}; rerun with --overwrite.")
        if len(values) != args.expected_source_samples:
            warnings += 1
            print(
                f"Warning: {recording.source} has {len(values)} samples; expected {args.expected_source_samples}.",
                file=sys.stderr,
            )

        resampled = resample_rows(values, target_samples)
        write_csv(native_path, values, recording.recording_timestamp, args.source_rate_hz)
        write_csv(resampled_path, resampled, recording.recording_timestamp, args.target_rate_hz)
        manifest_rows.append(
            {
                "user_id": recording.user_id,
                "recording_timestamp_s": recording.recording_timestamp,
                "modality": args.modality,
                "source_path": str(recording.source),
                "native_csv_path": str(native_path),
                "resampled_csv_path": str(resampled_path),
                "source_samples": len(values),
                "resampled_samples": len(resampled),
                "sensor_clock_span": f"{values[-1, 0] - values[0, 0]:.6f}",
            }
        )
        if index % 1000 == 0 or index == len(recordings):
            print(f"Processed {index}/{len(recordings)} recordings")

    manifest_path = args.output_dir / f"{args.modality}_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {len(recordings)} recordings and {manifest_path} ({warnings} sample-count warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
