#include "lcutil.h"
#include "timestamp.h"

/* 

	Declarations
	
*/


struct Parms { 
  float cx;
  float cy;
} parms = {0.1, 0.1};

/* 

	GPU functions
	
*/

__global__ void iterator_gpu(const float * __restrict__ T_source, float * __restrict__ T_destination, const int NXPROB, const int NYPROB,struct Parms parms){

	const int ix = blockIdx.x * blockDim.x + threadIdx.x ;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y ;
	
	
	if (ix>0 && ix<NXPROB-1 && iy>0 && iy<NYPROB-1)
		*(T_destination+ix*NYPROB+iy) = *(T_source+ix*NYPROB+iy)  + 
                          			parms.cx * (*(T_source+(ix+1)*NYPROB+iy) +
                          			*(T_source+(ix-1)*NYPROB+iy) - 
                          			2.0 * *(T_source+ix*NYPROB+iy)) +
                          			parms.cy * (*(T_source+ix*NYPROB+iy+1) +
                         			*(T_source+ix*NYPROB+iy-1) - 
                          			2.0 * *(T_source+ix*NYPROB+iy));
	

}

extern float Iterator_GPU(float* u, int NXPROB, int NYPROB, int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int STEPS){

	// --- GPU temperature distribution
	float *d_u_z_0;
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_u_z_0,  NXPROB * NYPROB * sizeof(float)) );
	float *d_u_z_1; 
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_u_z_1,  NXPROB * NYPROB * sizeof(float)) );
	
	// --- Grid size
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid (FRACTION_CEILING(NXPROB, BLOCK_SIZE_X), FRACTION_CEILING(NYPROB, BLOCK_SIZE_Y));
	
	cudaEvent_t start, stop;
	CUDA_SAFE_CALL( cudaEventCreate(&start) );
	CUDA_SAFE_CALL( cudaEventCreate(&stop) );
	
	
	CUDA_SAFE_CALL( cudaMemcpy(d_u_z_0, u, NXPROB * NYPROB * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_u_z_1, u+NXPROB*NYPROB, NXPROB * NYPROB * sizeof(float), cudaMemcpyHostToDevice) );
	
	
	CUDA_SAFE_CALL( cudaEventRecord(start) );
	// --- Jacobi iterations on the device
	for (int it=0; it<STEPS; it=it+2) {
		iterator_gpu<<<dimGrid, dimBlock>>>(d_u_z_0, d_u_z_1, NXPROB, NYPROB, parms);   // --- Update d_u_z_1     starting from data stored in d_u_z_0
		iterator_gpu<<<dimGrid, dimBlock>>>(d_u_z_1, d_u_z_0, NXPROB, NYPROB, parms);   // --- Update d_u_z_0     starting from data stored in d_u_z_1
	}      
	CUDA_SAFE_CALL( cudaEventRecord(stop) );

	// --- Copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy(u, d_u_z_0, NXPROB * NYPROB * sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float msecs = 0;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&msecs, start, stop) );
	
	// --- Release device memory
      	CUDA_SAFE_CALL(cudaFree(d_u_z_0));
      	CUDA_SAFE_CALL(cudaFree(d_u_z_1));
      	
      	return msecs;

}

