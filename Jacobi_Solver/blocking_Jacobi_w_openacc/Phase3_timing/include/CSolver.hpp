#ifndef CSolver_HPP
#define CSolver_HPP

#include <sstream>
#include "CMesh.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <openacc.h>
#include <mpi.h>
#include "Parallel_CSimple_timer.hpp"

template <typename T>
class CSolver {
public:
  void jacobi(CMesh<T> &M, int my_rank, int comm_sz, const size_t &max_steps,
              const size_t &PrintInterval, size_t local_rows, int rem, MPI_Comm comm){

    size_t step{0};
    size_t i, j;
    size_t n = M.rows;
    size_t m = M.colm;
    
    int next_rank = (my_rank == comm_sz - 1) ? MPI_PROC_NULL : my_rank + 1;
    int prev_rank = (my_rank == 0) ? MPI_PROC_NULL : my_rank - 1;


    T* d_field = M.field.data();
    T* d_new_field = M.new_field.data();

    //start OpenACC data region 
    #pragma acc data copy(d_field[:m*n], d_new_field[:m*n])
    {
    while (step < max_steps) {
      /**
       * each process performs TWO MPI_Send_recv()s one with the PREVIOUS process
       * and another with the NEXT process.
      */
      {
        CSimple_timer t("comm",  my_rank, comm_sz, MPI_COMM_WORLD);
        //send to prev_rank, recv from next_rank
        #pragma acc host_data use_device(d_field)
        MPI_Sendrecv(&d_field[m + 1], m-2, MPI_DOUBLE, prev_rank, 1,
                    &d_field[(n-1)*m + 1], m-2, MPI_DOUBLE, next_rank, 1, comm, MPI_STATUS_IGNORE);
      
        //send to next_rank, recv from prev_rank
        #pragma acc host_data use_device(d_field)
        MPI_Sendrecv(&d_field[(n-2)*(m) + 1], m-2, MPI_DOUBLE, next_rank, 0,
                     &d_field[1], m - 2, MPI_DOUBLE, prev_rank, 0, comm, MPI_STATUS_IGNORE);

      }

      {
        CSimple_timer t("comp", my_rank, comm_sz, MPI_COMM_WORLD);
        #pragma acc parallel loop gang
        for (i = 1; i < n - 1 ; i++) {
          #pragma acc loop vector
          for (j = 1; j < m - 1; j++) {
            d_new_field[i * m + j] =
                0.25*(d_field[(i + 1) * m + j] + d_field[(i - 1) * m + j] +
                        d_field[i * m + j + 1] + d_field[i * m + j - 1]);
         }
        }
      }
      //Swap fields
      #pragma acc parallel loop
      for (size_t i = 0; i < n * m; i++){ 
	      d_field[i] = d_new_field[i];
      }

      step++;
      
    }//while

    }//end data region
  }//jacobi
};//CSolver

#endif
