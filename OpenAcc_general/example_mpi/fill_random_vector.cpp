#include<curand.h>
int  fill_random_vector(double* buff, size_t N, long seed){
  curandGenerator_t gen;
  auto status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  status = curandSetPseudoRandomGeneratorSeed(gen, seed); 
#pragma acc data present_or_copyout(buff[:N]) 
  {
#pragma acc host_data use_device(buff) 
    {
      status = curandGenerateUniformDouble(gen, buff, N); 
    }
  }
  return status == CURAND_STATUS_SUCCESS;
}
