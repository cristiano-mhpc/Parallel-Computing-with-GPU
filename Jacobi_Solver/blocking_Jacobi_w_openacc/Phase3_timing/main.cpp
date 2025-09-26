#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "CMesh.hpp"
#include "CSolver.hpp"
#include <mpi.h>
#include "Parallel_CSimple_timer.hpp"

const size_t max_iter = 100;
const size_t PrintInterval = 2;

int main(int argc, char** argv){
    int comm_sz, my_rank;

  
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int num_of_gpus = acc_get_num_devices(acc_device_nvidida);
    acc_set_device_num(my_rank%comm_sz, acc_device_nvidia);
    
    //the size of the grid is (N+2)by(N+2)
    size_t N = std::stoi(argv[1]);

    //calculate the local number of rows
    size_t local_rows = (N+2)/comm_sz;

    //compute the remainder 
    int rem = (static_cast<int> (N+2) )%comm_sz;

    //determine the number of rows this process will handle
    size_t rows_to_handle = (my_rank < rem) ? local_rows + 1 : local_rows;

    //determine the global starting row for this process
      size_t start_row = (my_rank < rem)? my_rank*rows_to_handle + 0 : my_rank*rows_to_handle + rem;

  //Instantiate a mesh class for this process
    if (my_rank == 0 || my_rank == comm_sz -1){
      //process assigned to the head or tail of the 
      CMesh<double> my_data(rows_to_handle + 1, N+2, N+2, my_rank, comm_sz, start_row, MPI_COMM_WORLD);
      CSolver<double> solver;
      solver.jacobi(my_data, my_rank, comm_sz, max_iter, PrintInterval, local_rows, rem, MPI_COMM_WORLD);
    } else {
      //processes asigned to the body
      CMesh<double> my_data(rows_to_handle + 2, N+2, N+2, my_rank, comm_sz, start_row, MPI_COMM_WORLD);
      CSolver<double> solver;
      solver.jacobi(my_data, my_rank, comm_sz, max_iter, PrintInterval, local_rows, rem, MPI_COMM_WORLD);
    }


  CSimple_timer::print_timing_results(my_rank, comm_sz, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0; 

}
