# Cannon’s Algorithm for Distributed Matrix Multiplication (MPI + CBLAS)

This repository contains a C++ implementation of **Cannon’s algorithm** for dense
matrix multiplication using a **2‑D MPI Cartesian process grid** and
**CBLAS (OpenBLAS)** for the local block GEMM. Runtime measurement is provided by
a lightweight MPI‑aware utility `Parallel_CSimple_timer.hpp`.

> ✅ This version targets CPU nodes with BLAS acceleration (OpenBLAS/MKL).  
> If you are looking for the CUDA+MPI variant, see the GPU version in your other repo.

---

## Contents

```
.
├── main.cpp                     # Cannon’s algorithm (MPI + CBLAS)
├── Parallel_CSimple_timer.hpp   # MPI-aware timing helper (Total/comp ranges)
├── compile.sh                   # Build script (mpic++ + OpenBLAS)
├── run.sh                       # Example run (16 ranks, N=1000)
└── README.md                    # This file
```

---

## Build

The project is set up to compile with **mpic++** and link against **OpenBLAS**.
A convenience script is provided:

```bash
./compile.sh
```

The script uses:  
```bash
mpic++ main.cpp -o cart.x -lopenblas -O3 -Wall -march=native
```
If OpenBLAS is not in a default path, set `LIBRARY_PATH`/`LD_LIBRARY_PATH` or add
`-L/path/to/openblas -I/path/to/openblas/include` before running the script.

**Dependencies**
- An MPI implementation (Open MPI, MPICH, Intel MPI, etc.)
- CBLAS headers and OpenBLAS (or another BLAS that provides CBLAS)
- A C++17‑capable compiler

---

## Run

A sample launcher is included:

```bash
./run.sh
# which does, by default:
# mpirun -np 16 ./cart.x 1000
```

You may change the two key parameters:
- **Number of MPI ranks** (`-np P`) — **must be a perfect square** (1, 4, 9, 16, …)
- **Global matrix size** (`N`) — **must be divisible by sqrt(P)**

**Examples**
```bash
mpirun -np 4   ./cart.x 4096
mpirun -np 16  ./cart.x 8192
mpirun -np 36  ./cart.x 12000
```

---

## What the program does

At startup, the program validates the parallel geometry and inputs:

- Checks that `P` (total ranks) is a **perfect square**.
- Checks that `N % sqrt(P) == 0` so each rank gets an `N_loc = N / sqrt(P)` square block.

It then constructs a **2‑D periodic Cartesian communicator** and performs Cannon’s
initial skewing:

1. Shift each `A` block left by its **row coordinate**.
2. Shift each `B` block up by its **column coordinate**.

After the initial alignment, the algorithm performs `sqrt(P)` **iterate–multiply–shift**
steps:

- Local multiply: `C_loc += A_loc * B_loc` (via `cblas_dgemm`).
- Shift `A` left by 1, `B` up by 1 (toroidal neighbor exchanges with `MPI_Sendrecv`).

Local storage uses `std::vector<double>` for `A_loc`, `B_loc`, and `C_loc` with row‑major
layout expected by CBLAS.

---

## Timing & Output

Timing is instrumented with **CSimple_timer** ranges:
- `"Total"` — total time for the initial skew or the main iteration loop
- `"comp"`  — the time spent in the local `dgemm` (compute phase)

At the end, the program calls:
```cpp
CSimple_timer::print_timing_results(my_rank, comm_sz, MPI_COMM_WORLD);
```
which prints a per‑rank breakdown and a **global aggregation** (min/avg/max) across ranks.

> Tip: redirect output to a file for later parsing/plotting:
> ```bash
> mpirun -np 16 ./cart.x 8000 | tee timing_16ranks_N8000.txt
> ```

---

## Reproducibility Notes

- Use a **square** process count only (e.g., 4, 9, 16, 25, …).
- Ensure `N` is **divisible** by `sqrt(P)`.
- Pin ranks/threads if using threaded BLAS to avoid oversubscription (e.g., set `OPENBLAS_NUM_THREADS=1`).

**Environment examples (bash):**
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

---

## Extending this code

- Replace OpenBLAS with MKL by linking `-lmkl_intel_lp64 -lmkl_core -lmkl_sequential`.
- Add non‑blocking exchanges (`MPI_Isend/Irecv`) to overlap comm/comp.
- Introduce block‑cyclic layouts or larger tiles per rank to improve BLAS efficiency.
- Add a gather and verify phase for a full end‑to‑end test on moderate N.

---

## Acknowledgments

- BLAS: **OpenBLAS**
- MPI: Open MPI / MPICH
- Timer: `Parallel_CSimple_timer.hpp` (simple MPI‑aware scoped timer)

