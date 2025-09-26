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
#include <utility>
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
    
    T* d_field = M.field.data();
    T* d_new_field = M.new_field.data();

    int next_rank = (my_rank == comm_sz - 1) ? MPI_PROC_NULL : my_rank + 1;
    int prev_rank = (my_rank == 0) ? MPI_PROC_NULL : my_rank - 1;

    //start OpenACC data region 
    #pragma acc data copy(d_field[:m*n], d_new_field[:m*n])
    {
    while (step < max_steps) {
      /**
       * each process performs two MPI_Send_recv()s one with the PREVIOUS process
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
      
      
      //#pragma acc wait
      #ifdef PRINT
      //print to file at multiples of PrintInterval
      if(step%PrintInterval == 0){

        //std::cout << step << std::endl;

        #pragma acc update self(d_new_field[:m*n])

        //#pragma acc wait
        
        std::ostringstream temp;
        temp << "./data/" << std::setw(5) << std::setfill('0') << step <<".dat";
        std::ofstream filevar; 
 
        filevar.open(temp.str(), std::ios::app);

        if (!my_rank){          
          //first print p0 block
          for(size_t i = 0; i < n-1; i++){
            for(size_t j = 0; j < m; j++){
	            filevar << d_new_field[i * m + j] << " "; 
            }
            filevar << std::endl;
          }
          
          //receive blocks from other processes
          for (int rank = 1; rank < comm_sz; rank++){
            //compute how many rows each rank will send. We are not receiving ghost cells.
            size_t rows_from_rank = (rank < rem) ? local_rows + 1: local_rows;
            M.print_field.resize(rows_from_rank*m);
            MPI_Recv(M.print_field.data(), rows_from_rank*m, MPI_DOUBLE, rank, 2, comm, MPI_STATUS_IGNORE);

            //append to the .txt file right after receiving
            for(size_t i = 0; i < rows_from_rank; i++){
              for(size_t j = 0; j < m; j++){
                filevar << M.print_field[i * m + j] << " "; 
              }
              filevar << std::endl;
            }
        }

        filevar.close();

        } else {
          //just compute the local rows here again instead of passing from main. We are not sending ghost cells.
          size_t rows_to_handle = (my_rank < rem) ? local_rows + 1: local_rows;
          #pragma acc host_data use_device(d_new_field)
          MPI_Send(d_new_field + m, rows_to_handle*m, MPI_DOUBLE, 0, 2, comm);
        }
      }
      #endif
      
      /*NOTE: This is not working.
      #pragma acc parallel num_gangs(1) num_workers(1) vector_length(1) present(d_field[:m*n], d_new_field[:m*n])
      {
          T *temp = d_field; 
          d_field = d_new_field;
          d_new_field = temp;
      }
      */
      
        //swap by copying
        #pragma acc parallel loop
        for (size_t i = 0; i < n * m; i++){ 
	        d_field[i] = d_new_field[i];
        }

      }//timing comp

      step++;

    }//while
    
   }//end data region

  }//jacobi

};//CSolver

#endif
