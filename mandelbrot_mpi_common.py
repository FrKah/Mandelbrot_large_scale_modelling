"""Shared row-wise Mandelbrot helpers for the MPI implementations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    chunk_size: int
    size: tuple[int, int]
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def parse_parameters(description: str) -> tuple[Parameters, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("chunk_size", nargs="?", type=int, default=10)
    parser.add_argument("size", nargs="?", default="1000x1000")
    parser.add_argument("xlim", nargs="?", default="-2.2:0.75")
    parser.add_argument("ylim", nargs="?", default="-1.3:1.3")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    width, height = (int(value) for value in args.size.split("x"))
    xlim = tuple(float(value) for value in args.xlim.split(":"))
    ylim = tuple(float(value) for value in args.ylim.split(":"))
    if args.chunk_size < 1 or width < 1 or height < 1:
        parser.error("chunk_size and image dimensions must be positive")
    if len(xlim) != 2 or len(ylim) != 2:
        parser.error("limits must have the form min:max")

    return Parameters(args.chunk_size, (width, height), xlim, ylim), args


def row_chunks(width: int, chunk_size: int, workers: int) -> list[list[tuple[int, int]]]:
    chunks = [
        (start, min(start + chunk_size, width))
        for start in range(0, width, chunk_size)
    ]
    assignments = [[] for _ in range(workers)]
    for index, chunk in enumerate(chunks):
        owner = min(index * workers // len(chunks), workers - 1)
        assignments[owner].append(chunk)
    return assignments


def calculate_rows(
    start: int, stop: int, parameters: Parameters
) -> np.ndarray:
    width, height = parameters.size
    del width
    image = np.zeros((stop - start, height), dtype=np.float64)
    xconst = (parameters.xlim[1] - parameters.xlim[0]) / parameters.size[0]
    yconst = (parameters.ylim[1] - parameters.ylim[0]) / height

    for local_x, x in enumerate(range(start, stop)):
        cx = complex(parameters.xlim[0] + x * xconst, 0)
        for y in range(height):
            c = cx + complex(0, parameters.ylim[0] + y * yconst)
            z = 0j
            for iteration in range(100):
                z = z * z + c
                if abs(z) > 2:
                    image[local_x, y] = iteration
                    break
    return image


def plot_image(image: np.ndarray, parameters: Parameters, filename: str) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    plt.imshow(
        image.T,
        extent=(*parameters.xlim, *parameters.ylim),
        origin="lower",
    )
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")
    plt.margins(0, 0)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()
