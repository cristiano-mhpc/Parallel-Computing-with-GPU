#include "Parallel_CSimple_timer.hpp"
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <random>
#include <string>
#include <vector>

bool isPerfectSquare(int num) {
  if (num < 0) {
    return false; // No perfect square for negative numbers in integer domain
  }

  int root = static_cast<int>(
      std::sqrt(num)); // Calculate the integer part of the square root
  return root * root ==
         num; // Check if squaring the root gives the original number
}

void fill(double *vec, size_t dim, int my_rank) {
  // Initialize random engine and distribution
  std::random_device rd;                       // Seed generator
  std::mt19937 gen(rd());                      // Mersenne Twister engine
  std::uniform_int_distribution<> dis(1, 100); // Range: [1, 100]

  // Populate the vector
  for (int i = 0; i < dim; i++) {
    // vec[i]=dis(gen);
    vec[i] = 7.156;
  }
  vec[dim - 1] = 2.506 / (my_rank + 1);
  vec[dim - 4] = 58.609;
  vec[dim - 7] = 3.1416;
  vec[dim - 23] = 7.8416;
}

int main(int argc, char **argv) {
  /**
   * We implement the canonical algorithm of block matrix multiplication using
   * A 2-D distribution of data to the processes. This is accomplised by
   * creating a 2d cartesian layout for mapping 2d blocks of data to the
   * processes.
   */

  int comm_sz, my_rank;
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int num_of_gpus, mydevice;
  // Get number of gpus
  cudaGetDeviceCount(&num_of_gpus);
  // set device for each rank
  cudaSetDevice(my_rank % num_of_gpus);
  // get device for checking purpose
  cudaGetDevice(&mydevice);

  /**
   * The dimensions of the Cartesian grid
   */
  if (!isPerfectSquare(comm_sz)) {
    std::cout << "The number of process must be a perfect square." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // get the size of the square matrix
  size_t N = std::stoi(argv[1]);
  // the square root of communicator size
  int sqrt_comm_size = static_cast<int>(std::sqrt(comm_sz));

  // check if the dimension is a multiple of the squareroot of comm_size
  if (N % sqrt_comm_size != 0) {
    std::cout << "Usage: The dimension of the square matrix must be a multiple "
                 "of the sqrt(comm_size)"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  //********* Set up the grid ********//

  // we have a 2d plane
  int ndims = 2;

  // MPI will decide the dimensions
  int dimensions[2] = {0, 0};

  // create the dimensions
  MPI_Dims_create(comm_sz, ndims, dimensions);

  // specify periodic along both direction
  int periodicity[2] = {1, 1};

  // handle for the cartesian communicator
  MPI_Comm cart_comm;

  // create the cartesian grid of process
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dimensions, periodicity, 1,
                  &cart_comm);

  // query my coordinate in the cartesian grid
  int coords[2];
  MPI_Cart_coords(cart_comm, my_rank, ndims, coords);

  // query the rank of my neighbors. Very handy when exchanging
  int up, down, left, right;

  //******** Initialize the blocks assigned to me ******//
  int N_loc = N / sqrt_comm_size;
  std::vector<double> A_loc(N_loc * N_loc);
  std::vector<double> B_loc(N_loc * N_loc);

  // host buffer to contain a copy of the result from GPU
  std::vector<double> C_loc(N_loc * N_loc);

  fill(A_loc.data(), A_loc.size(), my_rank);
  fill(B_loc.data(), B_loc.size(), my_rank);

#ifdef PRINT
  std::ostringstream temp_A;
  temp_A << "A" << coords[0] << "_" << coords[1] << ".dat";
  std::ofstream filevar_A;

  std::ostringstream temp_B;
  temp_B << "B" << coords[0] << "_" << coords[1] << ".dat";
  std::ofstream filevar_B;

  filevar_A.open(temp_A.str(), std::ios::app);
  filevar_B.open(temp_B.str(), std::ios::app);

  for (size_t i = 0; i < N_loc; i++) {
    for (size_t j = 0; j < N_loc; j++) {
      filevar_A << A_loc[i * N_loc + j] << " ";
      filevar_B << B_loc[i * N_loc + j] << " ";
    }
    filevar_A << std::endl;
    filevar_B << std::endl;
  }
#endif

  // declare and allocate device buffers
  double *d_A_loc, *d_B_loc, *d_C_loc, *d_temp_A_loc, *d_temp_B_loc, *d_temp;
  cudaMalloc((void **)&d_A_loc, N_loc * N_loc * sizeof(double));
  cudaMalloc((void **)&d_B_loc, N_loc * N_loc * sizeof(double));
  cudaMalloc((void **)&d_C_loc, N_loc * N_loc * sizeof(double));

  // temporary device buffers
  cudaMalloc((void **)&d_temp_A_loc, N_loc * N_loc * sizeof(double));
  cudaMalloc((void **)&d_temp_B_loc, N_loc * N_loc * sizeof(double));

  // a temporary buffer used for swapping
  cudaMalloc((void **)&d_temp, sizeof(double));

  {
    CSimple_timer t1("Total", my_rank, comm_sz, MPI_COMM_WORLD);

    // copy initialized data on the host to the device
    // launch a stream to overlap copying of data
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_A_loc, A_loc.data(), N_loc * N_loc * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpy(d_B_loc, B_loc.data(), N_loc * N_loc * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    MPI_Cart_shift(cart_comm, 1, coords[0], &left, &right);
    MPI_Sendrecv(d_A_loc, N_loc * N_loc, MPI_DOUBLE, left, 0, d_temp_A_loc,
                 N_loc * N_loc, MPI_DOUBLE, right, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    // A_loc.swap(temp_A_loc);
    d_temp = d_A_loc;
    d_A_loc = d_temp_A_loc;
    d_temp_A_loc = d_temp;
    // cudaDeviceSynchronize();

    // cudaDeviceSynchronize();

    MPI_Cart_shift(cart_comm, 0, coords[1], &up, &down);
    MPI_Sendrecv(d_B_loc, N_loc * N_loc, MPI_DOUBLE, up, 1, d_temp_B_loc,
                 N_loc * N_loc, MPI_DOUBLE, down, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    // B_loc.swap(temp_B_loc);
    d_temp = d_B_loc;
    d_B_loc = d_temp_B_loc;
    d_temp_B_loc = d_temp;
    // cudaDeviceSynchronize();

    // Initializations relevant for Cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0;
    double beta = 1.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_loc, N_loc,
                N_loc,                  // Dimensions
                &alpha,                 // alpha
                d_B_loc, N_loc,         // B and its leading dimension
                d_A_loc, N_loc,         // A and its leading dimension
                &beta, d_C_loc, N_loc); // C and its leading dimension

    // cudaDeviceSynchronize();
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    for (int k = 0; k < sqrt_comm_size - 1; k++) {

      MPI_Sendrecv(d_A_loc, N_loc * N_loc, MPI_DOUBLE, left, 0, d_temp_A_loc,
                   N_loc * N_loc, MPI_DOUBLE, right, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      // cudaDeviceSynchronize();

      // cudaMemcpy(d_A_loc, d_temp_A_loc, N_loc*N_loc*sizeof(double),
      // cudaMemcpyDeviceToDevice);
      
      //swap
      d_temp = d_A_loc;
      d_A_loc = d_temp_A_loc;
      d_temp_A_loc = d_temp;
      // cudaDeviceSynchronize();

      MPI_Sendrecv(d_B_loc, N_loc * N_loc, MPI_DOUBLE, up, 1, d_temp_B_loc,
                   N_loc * N_loc, MPI_DOUBLE, down, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

      // cudaDeviceSynchronize();

      //swap
      d_temp = d_B_loc;
      d_B_loc = d_temp_B_loc;
      d_temp_B_loc = d_temp;

      // cudaDeviceSynchronize();
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_loc, N_loc,
                  N_loc,                  // Dimensions
                  &alpha,                 // alpha
                  d_B_loc, N_loc,         // B and its leading dimension
                  d_A_loc, N_loc,         // A and its leading dimension
                  &beta, d_C_loc, N_loc); // C and its leading dimension
      // cudaDeviceSynchronize();
    }

    // copy the result to host
    cudaMemcpy(C_loc.data(), d_C_loc, N_loc * N_loc * sizeof(double),
               cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();
  } // total time

#ifdef PRINT
  std::ostringstream temp_C;
  temp_C << "C" << coords[0] << "_" << coords[1] << ".dat";
  std::ofstream filevar_C;

  filevar_C.open(temp_C.str(), std::ios::app);

  for (size_t i = 0; i < N_loc; i++) {
    for (size_t j = 0; j < N_loc; j++) {
      filevar_C << C_loc[i * N_loc + j] << " ";
    }

    filevar_C << std::endl;
  }
#endif

  /*Have process 0 report the timing*/
  CSimple_timer::print_timing_results(my_rank, comm_sz, MPI_COMM_WORLD);

  if (!my_rank) {

    double flops = 2.0 * N * N * N;
    double gflops = flops / 1e9;
    std::cout << "gflops: " << gflops << std::endl;
    std::cout << "size: " << N << std::endl;
  }

  // free allocated device buffers
  cudaFree(d_A_loc);
  cudaFree(d_B_loc);
  cudaFree(d_C_loc);
  cudaFree(d_temp_A_loc);
  cudaFree(d_temp_B_loc);

  MPI_Comm_free(&cart_comm);

  MPI_Finalize();

  return 0;
}
