"""Dynamic row-chunk MPI Mandelbrot with worker-request scheduling."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from mandelbrot_mpi_common import calculate_rows, parse_parameters, plot_image


READY = 30
WORK = 31
RESULT = 32
STOP = 33


def main() -> None:
    parameters, args = parse_parameters(__doc__ or "")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    root = 0

    comm.Barrier()
    started = MPI.Wtime()
    if ranks == 1:
        image = calculate_rows(0, parameters.size[0], parameters)
    elif rank == root:
        image = np.zeros(parameters.size, dtype=np.float64)
        chunks = [
            (start, min(start + parameters.chunk_size, parameters.size[0]))
            for start in range(0, parameters.size[0], parameters.chunk_size)
        ]
        next_chunk = 0
        active: dict[int, tuple[int, int]] = {}
        completed = 0

        for worker in range(1, ranks):
            if next_chunk < len(chunks):
                chunk = chunks[next_chunk]
                next_chunk += 1
                active[worker] = chunk
                comm.send(chunk, dest=worker, tag=WORK)
            else:
                comm.send(None, dest=worker, tag=STOP)

        while completed < len(chunks):
            status = MPI.Status()
            comm.Probe(source=MPI.ANY_SOURCE, tag=RESULT, status=status)
            worker = status.source
            start, stop = active[worker]
            result = np.empty((stop - start, parameters.size[1]), dtype=np.float64)
            comm.Recv(result, source=worker, tag=RESULT)
            image[start:stop] = result
            completed += 1

            comm.recv(source=worker, tag=READY)
            if next_chunk < len(chunks):
                chunk = chunks[next_chunk]
                next_chunk += 1
                active[worker] = chunk
                comm.send(chunk, dest=worker, tag=WORK)
            else:
                del active[worker]
                comm.send(None, dest=worker, tag=STOP)
    else:
        image = None
        while True:
            status = MPI.Status()
            chunk = comm.recv(source=root, tag=MPI.ANY_TAG, status=status)
            if status.tag == STOP:
                break
            start, stop = chunk
            result = calculate_rows(start, stop, parameters)
            comm.Send(result, dest=root, tag=RESULT)
            comm.send(None, dest=root, tag=READY)

    elapsed = comm.reduce(MPI.Wtime() - started, op=MPI.MAX, root=root)
    if rank == root:
        print(f"TIME method=dynamic ranks={ranks} seconds={elapsed:.9f}")
        if not args.no_plot:
            plot_image(image, parameters, "mandelbrot_dynamic.png")


if __name__ == "__main__":
    main()
