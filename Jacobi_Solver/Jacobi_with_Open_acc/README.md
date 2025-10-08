# Parallel Jacobi Solver (MPI + OpenACC)

This project implements a **parallel Jacobi iterative solver** for a 2-D Poisson-type problem using **MPI** for distributed memory parallelism and **OpenACC** for GPU acceleration.  
It supports execution on both **single-node and multi-node** systems, and includes timing utilities and Gnuplot scripts for performance analysis.

---

##  Directory Structure

```
├── build/                        # Compiled objects and binaries
│   ├── debug/
│   └── release/
│       └── main.o
├── compile.sh                    # Build script (NVHPC + MPI)
├── ex_rel.x                      # Executable (release build)
├── include/                      # Header-only modules
│   ├── CMesh.hpp                 # Mesh class for domain decomposition and halo exchange
│   ├── CSolver.hpp               # Jacobi solver class (MPI + OpenACC)
│   └── Parallel_CSimple_timer.hpp# MPI-aware timing utility
├── jobs/                         # SLURM job scripts for various node counts
│   ├── 1node.job
│   ├── 2node.job
│   ├── 4node.job
│   ├── 8node.job
│   ├── 16node.job
│   ├── 32node.job
│   └── 64node.job
├── job_out/                      # Standard output from job runs
│   ├── 1_node.out
│   ├── 2_node.out
│   ├── 4_node.out
│   ├── 8_node.out
│   ├── 16_node.out
│   ├── 32_node.out
│   └── 64_node.out
├── job_error/                    # Standard error from job runs
│   ├── 1_nod.err
│   ├── 2_nod.err
│   ├── 4_nod.err
│   ├── 8_nod.err
│   ├── 16_node.err
│   ├── 32_node.err
│   ├── 64_node.err
│   └── etc.
├── main.cpp                      # Driver code for Jacobi solver
├── Makefile                      # Alternative build system
└── timings/                      # Performance and timing results
    ├── 1node_timing/
    │   ├── 12k.txt
    │   ├── 12k.gp
    │   ├── fig.png
    │   └── fig2.png
    └── multiple_node_timing/
        ├── 12k.txt
        ├── 12k.gp
        └── fig2.png
```

---

##  Code Overview

### `main.cpp`
The main driver performs:
- MPI initialization and rank assignment
- One-GPU-per-rank binding via OpenACC
- Domain decomposition of an `(N+2) × (N+2)` grid
- Halo management for interior/boundary ranks
- Iterative Jacobi solver invocation
- Timing collection and reporting

Example snippet:
```cpp
MPI_Init(&argc, &argv);
acc_set_device_num(my_rank % comm_sz, acc_device_nvidia);

CMesh<double> mesh(...);
CSolver<double> solver;
solver.jacobi(mesh, my_rank, comm_sz, max_iter, PrintInterval, local_rows, rem, MPI_COMM_WORLD);

CSimple_timer::print_timing_results(my_rank, comm_sz, MPI_COMM_WORLD);
MPI_Finalize();
```

### `CMesh.hpp`
Defines the **Mesh** class responsible for:
- Allocating local subdomains per rank
- Managing halo regions and inter-rank communication
- Providing accessors for interior and boundary points

### `CSolver.hpp`
Implements the **Jacobi iterative scheme**:
- Parallelized via `#pragma acc parallel loop collapse(2)`
- Performs iterative stencil updates and halo exchanges
- Controlled by `max_iter` and `PrintInterval`

### `Parallel_CSimple_timer.hpp`
A lightweight, MPI-aware timing utility:
- Captures wall-clock times for different phases
- Reports per-rank and global averages
- Used to build timing tables and plots

---

##  Compilation

### With `compile.sh`
Run the build script (adjust module names as needed):
```bash
bash compile.sh
```

### Manual build example
```bash
module load nvhpc/25.3 hpcx-mpi/2.19
mpicxx -acc -Minfo=accel -O3 -std=c++17 -Iinclude main.cpp -o ex_rel.x
```

### Debug build
```bash
mpicxx -acc -g -O0 -std=c++17 -Iinclude main.cpp -o ex_dbg.x
```

---

##  Running on HPC (Leonardo Example)

Each SLURM job script corresponds to a specific node count.

Example (1 node, 4 GPUs):
```bash
#!/bin/bash
#SBATCH -A ICT25_MHPC_0
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -t 00:10:00
#SBATCH -J jacobi_1node

srun ./ex_rel.x 12000
```

Example (multi-node):
```bash
srun -n 64 ./ex_rel.x 12000
```
The command-line argument (`12000`) specifies the global grid size.

---

##  Performance Results

Performance was measured on both CPU and GPU versions across single and multiple nodes.

| Configuration | Description | Plot |
|----------------|-------------|------|
| Single-node    | GPU vs CPU performance comparison | [1node](timings/1node_timing/fig2.png) |
| Multi-node     | Strong scaling with increasing nodes | [multi](timings/multiple_node_timing/fig2.png) |

Plots were generated using the provided Gnuplot scripts (`*.gp`).

---

##  Observations

- OpenACC offload yields significant single-node speedups.
- MPI domain decomposition scales well up to tens of nodes.
- Communication overhead dominates beyond ~64 ranks.
- The timing utility enables reproducible performance tracking.

---

##  Requirements

- **C++17 compiler**
- **MPI** (e.g., HPC-X or OpenMPI)
- **NVIDIA HPC SDK** with OpenACC support
- **Gnuplot** for visualization

---

##  Author

**Christian Tica**  
MHPC 
ICTP, Trieste, Italy  

---

