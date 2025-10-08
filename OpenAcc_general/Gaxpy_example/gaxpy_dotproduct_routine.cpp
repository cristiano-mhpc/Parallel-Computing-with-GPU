#include<iostream>
#include<vector> 
#include<iomanip>
#include<curand.h>
using namespace std;


void write_matrix_block(vector<double> matrix, size_t blocksize, size_t N){
    cout << "first block of the matrix" << endl << fixed << setprecision(8); 
    for (size_t j = 0; j < blocksize; ++j){
      for(size_t i = 0; i<6; ++i){
        cout << setw(16) << matrix[j*N+i];
      }
      cout << endl;
    }
}

#pragma acc routine vector
double dotproduct(int dim, double* V1, int inc1, double* V2, int inc2){
  double somma=0.0;
#pragma acc loop reduction(+:somma) 
  for (int i =0; i< dim; ++i){
    somma +=  V1[i*inc1]*V2[i*inc2]; 
  }
  return somma;
}

void write_vector_start(vector<double> vec, size_t blocksize){
    cout << fixed << setprecision(8);
    for (int i=0; i< blocksize; ++i){
      cout << setw(12) << vec[i]; 
    }
    cout << endl;
}

int main(){
  // initialize the pseudo-random-number gererator of curand
  curandGenerator_t generator; 
  auto status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT); 
    if (status != CURAND_STATUS_SUCCESS){
      cerr << "Curand Create Generator failed"<< endl; 
      return 0;
    }
  status = curandSetPseudoRandomGeneratorSeed(generator, 241202);
  if (status != CURAND_STATUS_SUCCESS){
    cerr << "Curand set seed failed"<< endl; 
    return 0;
  }
  const size_t  N=10000, N2=N*N;
  vector<double>  A(N2), X(N), Y(N);
  double* A_data = A.data(); 
  double* X_data = X.data(); 
  double* Y_data = Y.data();
  fill(X.begin(), X.end(),0.); 
  fill(Y.begin(), Y.end(),0.); 
#pragma acc data create(A_data[:N2], X_data[:N]) copyout(Y_data[:N])  
  {
//fill X and Y with random numbers using curand 
#pragma acc host_data use_device(X_data,Y_data) 
     {
      auto statusX  = curandGenerateUniformDouble(generator, X_data, N); 
      auto statusY  = curandGenerateUniformDouble(generator, Y_data, N); 
     }   
#pragma acc update self(X_data[:10])
    write_vector_start(X, 6);
//generates matrix A 
#pragma acc parallel loop 
    for (size_t i=0; i < N2; ++i){
      A_data[i] = 1.0/(i%N+1 + i/N);
    }
#pragma acc update self(A_data[:N2])
    write_matrix_block(A, 6, N);
// perform y = AX + y 
#pragma acc parallel 
#pragma acc loop gang 
  for (int i=0;i<N;++i){
    Y_data[i] += dotproduct(N, A_data+i, N, X_data, 1);
  }

  }//ends data region  
  write_vector_start(Y, 6); 
}
