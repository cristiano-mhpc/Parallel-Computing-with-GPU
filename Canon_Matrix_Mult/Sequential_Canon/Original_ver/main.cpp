#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm> // For std::copy
#include <cblas.h>
#include "Parallel_CSimple_timer.hpp"

bool isPerfectSquare(int num) {
    if (num < 0) {
        return false;  // No perfect square for negative numbers in integer domain
    }

    int sqrroot = static_cast<int>(std::sqrt(num)); 
    return sqrroot * sqrroot == num;                   
}

int main(int argc, char** argv){
	/**
	 * We implement the canonical algorithm of block matrix multiplication using 
	 * A 2-D distribution of data to the processes. This is accomplised by 
	 * creating a 2d cartesian layout for mapping 2d blocks of data to the processes. 
	 */

	int comm_sz, my_rank;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
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

	//check if the dimension is a multiple of the squareroot of comm_size 
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

	//Implement the algorithm
	{
		CSimple_timer t1("Total", my_rank, comm_sz, MPI_COMM_WORLD);
		
		MPI_Cart_shift(cart_comm, 1, coords[0], &left, &right);
		/**
		* send my block of elements to my left, receive the block of 
		* elements from my right. We dont use the same buffer to send
		* and receive from. I discovered this is not safe, apparently.
		*/ 
		MPI_Sendrecv(A_loc.data(), N_loc*N_loc, MPI_DOUBLE, left, 0, \
						temp_A_loc.data(), N_loc*N_loc, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		//copy data from temp_A_loc to A_loc	
		std::copy(temp_A_loc.begin(), temp_A_loc.end(), A_loc.begin());

		/**
		 * Shift my block of B to the process j step to up from me.
		 * j is the j-coordinate of the process in the grid. This is coods[1].
		*/ 

		/**
		 * query who is my up and down j steps from me
		*/
		MPI_Cart_shift(cart_comm, 0, coords[1], &up, &down);

		MPI_Sendrecv(B_loc.data(), N_loc*N_loc, MPI_DOUBLE, up, 1, \
					temp_B_loc.data(), N_loc*N_loc, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		//copy data from temp_B_loc to B_loc	
		std::copy(temp_B_loc.begin(), temp_B_loc.end(), B_loc.begin());

		{
			CSimple_timer t2("comp", my_rank, comm_sz, MPI_COMM_WORLD);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_loc, N_loc, N_loc,\
				1, A_loc.data(), N_loc, B_loc.data(), N_loc, 1, C_loc.data(), N_loc);
		}

	}//CSimple_timer t1

	{
		CSimple_timer t1("Total", my_rank, comm_sz, MPI_COMM_WORLD);

		for (int k=0; k < sqrt_comm_size - 1; k++){

			MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
		
			MPI_Sendrecv(A_loc.data(), N_loc*N_loc, MPI_DOUBLE, left, 0, \
						temp_A_loc.data(), N_loc*N_loc, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			//copy data from temp_A_loc to A_loc	
			std::copy(temp_A_loc.begin(), temp_A_loc.end(), A_loc.begin());

			/**
	 		* Now for the blocks of B. Query who is my up and down 1 step from me. Then send my block
			* of B to my up and receive from my down.
			*/
			MPI_Cart_shift(cart_comm, 0, 1, &up, &down);

			MPI_Sendrecv(B_loc.data(), N_loc*N_loc, MPI_DOUBLE, up, 1, \
						temp_B_loc.data(), N_loc*N_loc, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
			//copy data from temp_B_loc to B_loc	
			std::copy(temp_B_loc.begin(), temp_B_loc.end(), B_loc.begin());

			/**
			 * Call dgemm to perform C_loc = alpha * A_loc * B_loc + beta * C_loc, where alpha = beta = 1
			*/
			{	
				CSimple_timer t2("comp", my_rank, comm_sz, MPI_COMM_WORLD);
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_loc, N_loc, N_loc,\
					 	1, A_loc.data(), N_loc, B_loc.data(), N_loc, 1, C_loc.data(), N_loc);
			}//CSimple_timer t2
		}//for
	}//CSimple_timer t1

	MPI_Comm_free(&cart_comm);

	CSimple_timer::print_timing_results(my_rank, comm_sz, MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;

}
