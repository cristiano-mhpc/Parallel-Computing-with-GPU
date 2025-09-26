#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "CMesh.hpp"
#include "CSolver.hpp"
#include <openacc.h>
#include <mpi.h>

const size_t max_iter=1000;
const size_t PrintInterval=25;
const size_t N=9;

int main(int argc, char** argv){
  
  int comm_sz, my_rank; 

  //the size of the grid is (N+2)by(N+2)
  //size_t N = std::stoi(argv[1]);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  //int num_of_gpus = acc_get_num_devices(acc_device_nvidida);
  //acc_set_device_num(my_rank%comm_sz, acc_device_nvidia);

  //calculate the local number of rows
  size_t local_rows = (N+2)/comm_sz;

  int rem = (static_cast<int> (N+2) )%comm_sz; // the ramainder rows

  //determine the number of rows this process will handle
  size_t rows_to_handle = (my_rank < rem) ? local_rows + 1 : local_rows;

  //determine the global starting row for this process
  size_t start_row = (my_rank < rem)? my_rank*rows_to_handle + 0 : my_rank*rows_to_handle + rem;

  //Instantiate a mesh class for this process
  if (my_rank == 0){
    //process assigned to the head
    CMesh<double> my_data(rows_to_handle + 1, N+2, N+2, my_rank, comm_sz, start_row, MPI_COMM_WORLD);
    //std::cout << "From rank " << my_rank << ": " << std::endl;
    //my_data.test_print();
    CSolver<double> solver;
    solver.jacobi(my_data, my_rank, comm_sz, max_iter, PrintInterval, local_rows, rem, MPI_COMM_WORLD);
  
  } else if ( my_rank == comm_sz -1){
    //process assigned to the tail
    CMesh<double> my_data(rows_to_handle + 1, N+2, N+2, my_rank, comm_sz, start_row, MPI_COMM_WORLD);
    //std::cout << "From rank " << my_rank << ": " << std::endl;
    //my_data.test_print();
    CSolver<double> solver;
    solver.jacobi(my_data, my_rank, comm_sz, max_iter, PrintInterval, local_rows, rem, MPI_COMM_WORLD);

  } else {
    //processes asigned to the body
    CMesh<double> my_data(rows_to_handle + 2, N+2, N+2, my_rank, comm_sz, start_row, MPI_COMM_WORLD);
    //std::cout << "From rank " << my_rank << ": " << std::endl;
    //my_data.test_print();
    CSolver<double> solver;
    solver.jacobi(my_data, my_rank, comm_sz, max_iter, PrintInterval, local_rows, rem, MPI_COMM_WORLD);
  }
  
  //CSimple_timer::print_timing_results(size);

  MPI_Finalize();

  return 0; 

}
