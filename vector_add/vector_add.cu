#include <stdio.h>
//#include <cuda.h>

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512

__global__ void Vec_add(
		const float dev_a[] /* in */,
		const float dev_b[] /* in */,
		float       dev_c[] /* out */, 
		const int   n   /* in */ ){

	int global_index = blockDim.x * blockIdx.x + threadIdx.x;

	/* total threads = blk_ct *th_per_blk may be > n */
	if (global_index < n)
		dev_c[global_index] = dev_a[global_index] + dev_b[global_index];

} /*Vec_add*/


void Allocate_vectors(
		float** d_x_p  /*out*/,
		float** d_y_p  /*out*/,
		float** d_z_p  /*out*/,
		float** h_x_p /*out*/,
		float** h_y_p /*out*/,
		float** h_z_p /*out*/,

		float** z_copy /*out*/,
		int n        /* in */ ) {

	/* x ,y , and z are used on host and device */
	cudaMalloc(d_x_p, n*sizeof(float));
	cudaMalloc(d_y_p, n*sizeof(float));
	cudaMalloc(d_z_p, n*sizeof(float));

	/*cz is only used on host*/
	*h_x_p = (float*)malloc(n*sizeof(float));
	*h_y_p = (float*)malloc(n*sizeof(float));
	*h_z_p = (float*)malloc(n*sizeof(float));

	/*A copy of the d_z*/
    *z_copy = (float*)malloc(n*sizeof(float));

}/*Allocate_vectors*/
      
void Serial_vec_add(
		const float x[],
		const float y[],
		float       cz[],
		const int   n) {
	for (int i =0; i <n; i++)
		cz[i] = x[i] + y[i];

}/* Serial_vec_add */


double Two_norm_diff(
		const float z[],
		const float cz[], 
		const int   n){

	double diff, sum = 0.0;

	for (int i =0; i << n; i++){
		diff = z[i] - cz[i];
		sum += diff*diff;
	}

	return sqrt(sum);

} /* Two_norm_diff */

void Free_vectors(
		float* d_x,
		float* d_y,
		float* d_z,
        float* h_x,
		float* h_y,
		float* h_z,

		float* copy_z /*in/out*/){

	/*Allocated with cudaMalloc*/
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	/*Allocated with malloc */
	free(h_x);
	free(h_y);
	free(h_z);
  
	free(copy_z);

}/*Free_vectors*/



int main(int argc, char* argv[]){

	float *d_x, *d_y, *d_z, *h_x, *h_y, *h_z,*copy_z;
	double diff_norm; 

    int size = N*sizeof(float);

	Allocate_vectors(&d_x, &d_y, &d_z, &h_x, &h_y, &h_z, &copy_z, N);

	/*Initialize the vectors*/
	for (int i = 0; i <N; i++){
		h_x[i] = rand()/(float)RAND_MAX;
		h_y[i] = rand()/(float)RAND_MAX; 
	}

	/*copy the host arrays to corresponding device arrays*/
	cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

	/*Invoke kernel and wait for it to complete*/
	Vec_add <<<ceil(N/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_x, d_y, d_z, N);

	cudaMemcpy(copy_z, d_z, size, cudaMemcpyDeviceToHost);

	/*Check the cpu_version of the vector_add*/
	Serial_vec_add(h_x, h_y, h_z, N);

	/*sync host with device*/
	cudaDeviceSynchronize();
    
	/*get the error */
	diff_norm = Two_norm_diff(h_z, copy_z, N);

	printf("Two-norm of difference between host and ");

	printf("device = %e\n", diff_norm); 

	/*Free storage and quit*/
	Free_vectors(d_x, d_y, d_z, h_x, h_y, h_z ,copy_z);

	return 0; 

} /*main*/

