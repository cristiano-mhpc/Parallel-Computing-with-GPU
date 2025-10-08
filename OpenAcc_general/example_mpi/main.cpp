#include<iostream>
#include<mpi.h> 
#include<openacc.h>
#include<curand.h>
#include<vector>
#include<iomanip>
using namespace std;

int  fill_random_vector(double* buff, size_t N, long seed); 

void write_vector_start(vector<double>, size_t blocksize); 

int main(int argc, char *argv[]){
  const size_t N=100000; 
  MPI_Init(&argc, &argv); 
  int rank;
  int mpisize; 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize); 
  int num_of_gpus = acc_get_num_devices(acc_device_nvidia); 
  acc_set_device_num(rank%num_of_gpus, acc_device_nvidia); 
  int mydevice = acc_get_device_num(acc_device_nvidia); 
  cout << "hello, rank " << rank << "  is using device " << mydevice << endl; 
  
  vector<double>  myvec(N); 
  double* mybuff = myvec.data(); 
  double start_time = MPI_Wtime(); 
#pragma acc enter data create(mybuff[:N])
  long seed = 12345 * (rank+5);
  fill_random_vector(mybuff, N, seed); 
  if (rank == 0){
  #pragma acc update self(mybuff[:10]) 
  write_vector_start(myvec, 10); 	  
  }
 
    if (rank == 2) {
      #pragma acc host_data use_device(mybuff) 
      MPI_Reduce(MPI_IN_PLACE, mybuff, N, MPI_DOUBLE, MPI_SUM, 2, MPI_COMM_WORLD);
    } else {
	#pragma acc host_data use_device(mybuff) 
        MPI_Reduce(mybuff, nullptr, N, MPI_DOUBLE, MPI_SUM, 2, MPI_COMM_WORLD); 
    } 
  if (rank == 2){
    #pragma acc update self(mybuff[:N]) 
    write_vector_start(myvec, 10);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  if (rank == 0){
    cout << setprecision(6) << setw(10) << end_time - start_time << endl;
  }
#pragma acc exit data delete(mybuff)
  MPI_Finalize(); 
  return 0;  
}
