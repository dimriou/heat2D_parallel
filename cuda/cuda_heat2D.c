/****************************************************************************
 * FILE: cuda_heat2D.c
 * DESCRIPTIONS:  
 *   HEAT2D Example - Parallelized C Version for nVidia's CUDA acceleration
 
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


#define NXPROB      	6000  	/* x dimension of problem grid */
#define NYPROB      	6000  	/* y dimension of problem grid */
#define STEPS       	3000   	/* number of time steps */
#define BLOCK_SIZE_X 	8	/* Block size (x-dimension) */
#define BLOCK_SIZE_Y 	8	/* Block size (y-dimension)  */

extern float Iterator_GPU(float* , int , int , int , int , int );

int main (int argc, char *argv[])
{

    
void inidat(int,int,float*), prtdat(int,int,float*,char*);

static float  u[2][NXPROB][NYPROB];     /* array for grid */

float time;              		/* timing variables */

     
      printf ("Starting cuda_heat2D.\n");
      size_t freeCUDAMem, totalCUDAMem;
      cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
      printf("Total GPU memory %ld, free %ld\n", totalCUDAMem, freeCUDAMem);

      /* Initialize grid */
      printf("Grid size: X= %d  Y= %d  Time steps= %d\n",NXPROB,NYPROB,STEPS);
      printf("Initializing grid and writing initial.dat file...\n");
      printf("Block size: %d x %d ...\n",BLOCK_SIZE_X,BLOCK_SIZE_Y);
      inidat(NXPROB, NYPROB, &u[0][0][0]);
      //prtdat(NXPROB, NYPROB, &u[0][0][0], "initial.dat");
     
      
      time = Iterator_GPU(&u[0][0][0],NXPROB,NYPROB,BLOCK_SIZE_X,BLOCK_SIZE_Y,STEPS);

	
      printf ( "Time = %.2f milliseconds (%.2f seconds), Bandwidth: %.2f GB/sec\n",time, time/1000.0f,
      								(8*NXPROB*NYPROB*sizeof(float))/
      								(time/1000.0f/STEPS)/
      								(float)(1024*1024*1024));
    	
      /* Write final output, call X graph and finalize MPI */
      printf("Writing final.dat file...\n");
      //prtdat(NXPROB, NYPROB, &u[0][0][0], "final.dat");
      printf("Sample value : %6.1f \n",u[0][1][1]);
      printf("EXITING...\n");
      
}   

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float *u) {
int ix, iy;

for (ix = 0; ix <= nx-1; ix++) 
  for (iy = 0; iy <= ny-1; iy++)
     *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
     

for (ix=0; ix<nx; ix++) 
  for (iy=0; iy<ny; iy++) 
     *(u+nx*ny+ix*ny+iy) = 0.0;    
     
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
for (iy = ny-1; iy >= 0; iy--) {
  for (ix = 0; ix <= nx-1; ix++) {
    fprintf(fp, "%6.1f", *(u1+ix*ny+iy));
    if (ix != nx-1) 
      fprintf(fp, " ");
    else
      fprintf(fp, "\n");
    }
  }
fclose(fp);
}
