#include <mpi.h>
#include <iostream>
#include <vector>
#include <string> //#include "Parallel_CSimple_timer.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <openacc.h>



bool isPerfectSquare(int num) {
    if (num < 0) {
        return false;  // No perfect square for negative numbers in integer domain
    }

    int root = static_cast<int>(std::sqrt(num)); // Calculate the integer part of the square root
    return root * root == num;                   // Check if squaring the root gives the original number
}


void matMul(int m, int n, int k, double* A, double* B, double* C){

  cublasHandle_t handle;
  cublasCreate(&handle);

  double alpha = 1.0;
  double beta = 1.0;

  //data region. Copy matrices A and B into the device and copy C back to the host
  #pragma acc data copyin(A[0:m*k], B[0:k*n]) copy(C[0:m*n])
  {
	//device pointers
    double* d_A = (double *)acc_deviceptr(A);
    double* d_B = (double *)acc_deviceptr(B);
    double* d_C = (double *)acc_deviceptr(C);

	cublasDgemm(handle, 
              CUBLAS_OP_T,CUBLAS_OP_T, 
			  m ,n ,k,               //Dimensions
			  &alpha,               
			  d_A, m,                //A and its leading dimension
			  d_B, k,                //B and its leading dimension
			  &beta,                 
			  d_C, m);               //C and its leading dimension

  }

  //destory cublas handle 
  cublasDestroy(handle);

}

int main(int argc, char** argv){
	/**
	 * We implement the Cannon's algorithm of block matrix multiplication using 
	 * A 2-D distribution of data to the processes. This is accomplised by 
	 * creating a 2d cartesian layout for mapping 2d blocks of data to the processes. 
	 */

	int comm_sz, my_rank;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int num_of_gpus = acc_get_num_devices(acc_device_nvidia);
    acc_set_device_num(my_rank%comm_sz, acc_device_nvidia);	

    //cublas handle
	cublasHandle_t handle;
    cublasCreate(&handle);
	//relevant later for calling cublas 
	double alpha = 1.0;
    double beta = 1.0;

	/**
	 * The dimensions of the Cartesian grid
	*/
	if ( !isPerfectSquare(comm_sz) ){
		std::cout << "The number of process must be a perfect square." << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//get the size of the square matrix
	size_t N = std::stoi(argv[1]);
	//the square root of communicator size
	int sqrt_comm_size = static_cast<int>(std::sqrt(comm_sz));

	//check if the dimension is a multiple of the square root of comm_size 
	if (N%sqrt_comm_size != 0){
		std::cout << "Usage: The dimension of the square matrix must be a multiple of the sqrt(comm_size)" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);

	}

	//********* Set up the grid ********//

	//we have a 2d plane
	int ndims = 2;

	//MPI will decide the dimensions 
	int dimensions[2] = {0, 0};

	//create the dimensions
	MPI_Dims_create(comm_sz, ndims, dimensions);

	//specify periodic along both direction
	int periodicity[2] = {1,1};

	//handle for the cartesian communicator
	MPI_Comm cart_comm;

	//create the cartesian grid of process
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dimensions, periodicity, 1, &cart_comm);

	//query my coordinate in the cartesian grid
	int coords[2];
	MPI_Cart_coords(cart_comm, my_rank, ndims, coords);

	//query the rank of my neighbors. Very handy when exchanging
	int up, down, left, right;

	//******** Initialize the blocks assigned to me ******//
	int N_loc = N/sqrt_comm_size;

	std::vector<double> A_loc(N_loc*N_loc, 1.0);
	std::vector<double> B_loc(N_loc*N_loc, 1.0);
	std::vector<double> C_loc(N_loc*N_loc);

	//temporary containers so we dont use the same buffer in the Sendrecv exchange later
	std::vector<double> temp_B_loc(N_loc*N_loc);
	std::vector<double> temp_A_loc(N_loc*N_loc);
     
	//declare device pointers
	double* d_A_loc = A_loc.data();
	double* d_B_loc = B_loc.data();
	double* d_C_loc = C_loc.data();

    double* d_temp_A_loc = temp_A_loc.data();
	double* d_temp_B_loc = temp_B_loc.data();
    
	//todo: Maybe try this instead?
	//double* d_temp_A_loc = (double *)acc_deviceptr(temp_A_loc);
 
    
    // copyin all vactors in later in the timing 
    #pragma acc data copyin(d_A_loc[:N_loc*N_loc], d_B_loc[:N_loc*N_loc], d_C_loc[:N_loc*N_loc], d_temp_A_loc[:N_loc*N_loc], d_temp_B_loc[:N_loc*N_loc], d_C_loc[:N_loc*N_loc])
    {
	  {
      CSimple_timer t1("Total", my_rank, comm_sz, MPI_COMM_WORLD);
	  MPI_Request request_send, request_recv, request_send2, request_recv2;  
     
	  MPI_Cart_shift(cart_comm, 1, coords[0], &left, &right);

	  #pragma acc host_data use_device(d_A_loc,d_temp_A_loc)
	  MPI_Sendrecv(d_A_loc, N_loc*N_loc, MPI_DOUBLE, left, 0, \
					d_temp_A_loc, N_loc*N_loc, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  //A_loc.swap(temp_A_loc);
	  //todo change this with the one similar to Johns
	  #pragma acc parallel loop
      for (size_t i = 0; i < N_loc*N_loc; i++){ 
	      d_A_loc[i] = d_temp_A_loc[i];
      }

	  MPI_Cart_shift(cart_comm, 0, coords[1], &up, &down);

	  #pragma acc host_data use_device(d_B_loc,d_temp_B_loc)
	  MPI_Sendrecv(d_B_loc, N_loc*N_loc, MPI_DOUBLE, up, 1, \
					d_temp_B_loc, N_loc*N_loc, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	
	  //B_loc.swap(temp_B_loc);
	  #pragma acc parallel loop
      for (size_t i = 0; i < N_loc*N_loc; i++){ 
	      d_B_loc[i] = d_temp_B_loc[i];
      }
    
	  MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
	  MPI_Cart_shift(cart_comm, 0, 1, &up, &down);

	  for (int k=0; k < sqrt_comm_size - 1; k++){
		    
			#pragma acc host_data use_device(d_A_loc,d_temp_A_loc)
			{
	        MPI_Isend(d_A_loc, N_loc*N_loc, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &request_send);
		    MPI_Irecv(d_temp_A_loc, N_loc*N_loc, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &request_recv);
			}

			#pragma acc host_data use_device(d_B_loc,d_temp_B_loc)
			{
		    MPI_Isend(d_B_loc, N_loc*N_loc, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &request_send2);
		    MPI_Irecv(d_temp_B_loc, N_loc*N_loc, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &request_recv2);
			}

            {
	          CSimple_timer t2("comp", my_rank, comm_sz, MPI_COMM_WORLD);		
              //matMul(N_loc, N_loc, N_loc, A_loc.data(), B_loc.data(),C_loc.data());
			  cublasDgemm(handle, 
                         CUBLAS_OP_T,CUBLAS_OP_T, 
			             N_loc, N_loc, N_loc,             //Dimensions
			             &alpha,                         //alpha
			             d_B_loc, N_loc,                //A and its leading dimension
			             d_A_loc, N_loc,                //B and its leading dimension
			             &beta,                 
			             d_C_loc, N_loc);               //C and its leading dimension
	        }

		    MPI_Wait(&request_recv, MPI_STATUS_IGNORE);
		    MPI_Wait(&request_send, MPI_STATUS_IGNORE);

		    //A_loc.swap(temp_A_loc);
			#pragma acc parallel loop
            for (size_t i = 0; i < N_loc*N_loc; i++){ 
	           d_A_loc[i] = d_temp_A_loc[i];
            }

		    MPI_Wait(&request_recv2, MPI_STATUS_IGNORE);
		    MPI_Wait(&request_send2, MPI_STATUS_IGNORE);

		    //B_loc.swap(temp_B_loc);
			#pragma acc parallel loop
            for (size_t i = 0; i < N_loc*N_loc; i++){ 
	          d_B_loc[i] = d_temp_B_loc[i];
            }

	  }
            
	  {
		    CSimple_timer t2("comp", my_rank, comm_sz, MPI_COMM_WORLD);    
            //matMul(N_loc, N_loc, N_loc, A_loc.data(), B_loc.data(),C_loc.data());
			cublasDgemm(handle, 
              CUBLAS_OP_T,CUBLAS_OP_T, 
			  N_loc, N_loc, N_loc,                   //Dimensions
			  &alpha,                                //alpha
			  d_B_loc, N_loc,                        //A and its leading dimension
			  d_A_loc, N_loc,                        //B and its leading dimension
			  &beta,                 
			  d_C_loc, N_loc);                       //C and its leading dimension
	  }


	  }//CSimple_timer_total


	}//end data_region

    std::cout << "From Rank: " << my_rank << std::endl;
	for (int i = 0; i < N_loc; i++){
		for (int j = 0; j < N_loc; j++){
			std::cout << C_loc[i * N_loc + j] << " ";
		}
		std::cout << std::endl;
	}

	long long int comp_time = CSimple_timer::print_timing_results(my_rank, comm_sz, MPI_COMM_WORLD);	
	if (!my_rank){

	  long long int comp_time = CSimple_timer::print_timing_results(my_rank, comm_sz, MPI_COMM_WORLD);

	  std::cout << "computation time in s: " << comp_time/1e6 << std::endl;
        
	  double flops = 2.0 * N * N * N;

	  double gflops = flops / comp_time / 1e9;
   
	  std::cout << "gflops: " << gflops << std::endl;

    }

    cublasDestroy(handle);

    MPI_Comm_free(&cart_comm);

	MPI_Finalize();

	return 0;

}
