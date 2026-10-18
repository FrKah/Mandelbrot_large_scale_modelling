"""Benchmark the dynamic row scheduler and create comparison plots."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


TIME_RE = re.compile(r"TIME method=dynamic ranks=(\d+) seconds=([0-9.eE+-]+)")


def parse_ranks(value: str) -> list[int]:
    try:
        ranks = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from error
    if not ranks or any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be positive")
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--size", default="5000x5000")
    parser.add_argument("--ranks", type=parse_ranks, default=parse_ranks("1,2,4,8,16"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--label", default="dynamic-one-node")
    parser.add_argument("--output-dir", type=Path, default=Path("Outputs"))
    parser.add_argument("--output-prefix", default="benchmark_dynamic_one_node")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.repetitions < 1:
        parser.error("--chunk-size and --repetitions must be positive")

    directory = Path(__file__).resolve().parent
    program = directory / "mandelbrot_mpi_dynamic.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, object]] = []

    for rank_count in args.ranks:
        for repetition in range(1, args.repetitions + 1):
            command = [
                args.mpiexec,
                "-n",
                str(rank_count),
                sys.executable,
                str(program),
                str(args.chunk_size),
                args.size,
                "--no-plot",
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{' '.join(command)} failed with exit code "
                    f"{completed.returncode}\n{completed.stderr}"
                )
            match = TIME_RE.search(completed.stdout)
            if match is None:
                raise RuntimeError(f"Missing timing output from {' '.join(command)}")
            measurements.append(
                {
                    "label": args.label,
                    "size": args.size,
                    "chunk_size": args.chunk_size,
                    "method": "dynamic",
                    "ranks": rank_count,
                    "repetition": repetition,
                    "seconds": float(match.group(2)),
                }
            )

    output_csv = args.output_dir / f"{args.output_prefix}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)

    means = {
        rank: sum(
            float(row["seconds"]) for row in measurements if row["ranks"] == rank
        )
        / sum(row["ranks"] == rank for row in measurements)
        for rank in args.ranks
    }
    baseline = means[args.ranks[0]]
    plots = [
        (
            "seconds",
            "Elapsed time (s)",
            "Dynamic Mandelbrot scheduling time",
            [means[rank] for rank in args.ranks],
        ),
        (
            "speedup",
            f"Speedup relative to {args.ranks[0]} rank",
            "Dynamic Mandelbrot scheduling speedup",
            [baseline / means[rank] for rank in args.ranks],
        ),
        (
            "efficiency",
            "Parallel efficiency",
            "Dynamic Mandelbrot scheduling efficiency",
            [baseline / means[rank] / rank for rank in args.ranks],
        ),
    ]
    for suffix, ylabel, title, values in plots:
        figure, axis = plt.subplots()
        axis.plot(args.ranks, values, marker="o", linewidth=2, label="dynamic")
        axis.set_xlabel("Number of MPI workers (ranks)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title} ({args.label}, {args.size}, chunk={args.chunk_size})")
        axis.set_xticks(args.ranks)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{args.output_prefix}_{suffix}.png", dpi=180)
        plt.close(figure)

    print(
        f"Wrote {output_csv} and {len(plots)} plots to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
