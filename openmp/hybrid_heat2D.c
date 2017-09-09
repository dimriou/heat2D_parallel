/****************************************************************************
* FILE: hybrid_heat2D.c
* DESCRIPTIONS:
*   HEAT2D Example - Parallelized C Version
*   This example is based on a simplified two-dimensional heat
*   equation domain decomposition.  The initial temperature is computed to be
*   high in the middle of the domain and zero at the boundaries.  The
*   boundaries are held at zero throughout the simulation.  During the
*   time-stepping, an array containing two domains is used; these domains
*   alternate between old data and new data.
*
*   In this parallelized version, the grid is decomposed by the master
*   process and then distributed by rows to the worker processes.  At each
*   time step, worker processes must exchange border data with neighbors,
*   because a grid point's current temperature depends upon it's previous
*   time step value plus the values of the neighboring grid points.  Upon
*   completion of all time steps, the worker processes return their results
*   to the master process.
*
*   Two data files are produced: an initial data set and a final data set.
* AUTHOR: Blaise Barney - adapted from D. Turner's serial C version. Converted
*   to MPI: George L. Gusciora (1/95)
* LAST REVISED: 04/02/05
****************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STEPS       3000             	/* number of time steps */
#define BEGIN       1                 	/* message tag */
#define LTAG        2                  	/* message tag */
#define RTAG        3                  	/* message tag */
#define TTAG        5                  	/* message tag */
#define DTAG        6                  	/* message tag */
#define NONE        0                  	/* indicates no neighbor */
#define DONE        4                  	/* message tag */
#define MASTER      0                  	/* taskid of first process */

/* Parallel program definitions*/
#define BLOCK_LENGTH   	1000		/* dimensions of one block (they are rectangles of size: BLOCK_LENGTH x BLOCK_LENGTH */
#define MIN_PROCESSES 	4	      	/* Minimum number of worker processes */
#define MAX_PROCESSES 	36	      	/* Maximum number of worker processes */
#define THREADS		8		/* Number of threads when using OpenMP parallelism, minimum 2 */
#define CONVERGENCE_CHK 0		/* Enable/disable convergence check */
#define INTERVAL	3500		/* Ater how many rounds are we checking for convergence */
#define SENSITIVITY	0.05		/* Convergence's sensitivity (EPSILON) */


struct Parms {
    float cx;
    float cy;
} parms = {0.1, 0.1};

int main ( int argc, char *argv[] )
{
    /* initialisation, printing, updating functions */
    void inidat(), prtdat(), update();
    
    /* Timing variables */
    double start_time, end_time,local_elapsed,elapsed;
    
    /* Block array */
    static float  u[2][BLOCK_LENGTH+2][BLOCK_LENGTH+2];	/* block (double buffered) */
    
    /* Other variables */
    int	taskid,                     	/* this task's unique id */
        dest, source,               	/* to - from for message send-receive */
        left,right,top,down,        	/* neighbor tasks */
        msgtype,                    	/* for message types */
        start_x,start_y,end_x,end_y,   	/* start-end positions when computing values in block */
        ix,iy,iz,it,        		/* loop variables */
        numworkers,numtasks,		/* number of workers and number of processes (tasks) */
        rc;				/* misc */

    /* First, find out my taskid and how many tasks are running */
    MPI_Init ( &argc,&argv );
    MPI_Comm_size ( MPI_COMM_WORLD,&numtasks );
    MPI_Comm_rank ( MPI_COMM_WORLD,&taskid );
    numworkers = numtasks-1;

    if ( taskid == MASTER ) {
        /************************* master code *******************************/
        /* Check if numworkers is within range - quit if not */
        if ( ( numworkers < MIN_PROCESSES ) || ( numworkers > MAX_PROCESSES ) ){
            printf ( "ERROR: the number of tasks must be between %d and %d.\n",
                     MIN_PROCESSES+1, MAX_PROCESSES+1 );
            printf ( "Quitting...\n" );
            MPI_Abort ( MPI_COMM_WORLD, rc );
            exit ( 1 );
        }
        /* Check if block is too small */
        if ( BLOCK_LENGTH*BLOCK_LENGTH < 16 ){
            printf ( "ERROR: block is too small.\n");
            printf ( "Quitting...\n" );
            MPI_Abort ( MPI_COMM_WORLD, rc );
            exit ( 1 );
        }
        /* Check if the array dimensions are equal (e.g. a 5x5 or 6x6 rectangle) */
        if ( sqrt(numworkers)!=floor(sqrt(numworkers)) ){
            printf ( "ERROR: the number of worker processes (thus all minus one) must have an integer square root (e.g."
            			"5x5 = 25 processes, Total: 26).\n");
            
            printf ( "Quitting...\n" );
            MPI_Abort ( MPI_COMM_WORLD, rc );
            exit ( 1 );
        }
        printf ( ">> Starting mpi_heat2D with %d worker tasks.\n", numworkers );
        printf ( ">> Block size: %d elements (%d x %d).\n", BLOCK_LENGTH*BLOCK_LENGTH, BLOCK_LENGTH, BLOCK_LENGTH  );
        printf ( ">> Grid size: %d elements (%.0f x %.0f).\n", BLOCK_LENGTH*BLOCK_LENGTH*numworkers, sqrt(BLOCK_LENGTH*BLOCK_LENGTH*numworkers),  sqrt(BLOCK_LENGTH*BLOCK_LENGTH*numworkers) );
        printf ( ">> Time steps= %d\n",STEPS );
        printf ( ">> Threads used (per process) = %d\n\n", THREADS);

	/* No need to synchronize, call MPI_Barrier because it's in the same communicator */
	MPI_Barrier(MPI_COMM_WORLD);
	
        /* Now we don't have to wait for results from worker tasks */
      
      	/* Matching second call of the workers */
        MPI_Barrier(MPI_COMM_WORLD);
        printf ( "\nEXITING.\n" );

    }   /* End of master code */



    /************************* workers code **********************************/
    if ( taskid != MASTER ) {
    
   	/* Square root of number of workers (i.e. how many workers fit in one dimension of the block grid) */
    	int ROOT_WORKERS = (int)sqrt(numworkers);
    	int uber = 0;
    	
    	/* Initialize everything - including the borders - to zero */
      	for (iz=0; iz<2; iz++)
          for (ix=0; ix<BLOCK_LENGTH+2; ix++) 
            for (iy=0; iy<BLOCK_LENGTH+2; iy++) 
               u[iz][ix][iy] = 0.0;
    	
        /* Initialize everything to random values */
      	inidat ( BLOCK_LENGTH+2, BLOCK_LENGTH+2, u, taskid, ROOT_WORKERS );      	
      	
      	/* Determine border elements.  Need to consider first and last columns. */
        /* Obviously, row 0 can't exchange with row 0-1.  Likewise, the last */
        /* row can't exchange with last+1.  */
        /* Also calculate neighbours (their taskid) */
        start_x = 1;
        top = taskid - ROOT_WORKERS;
        if ( taskid <= ROOT_WORKERS ) {
            start_x = 2;
            top = NONE;
       	}
       	
       	down = taskid + ROOT_WORKERS;
	end_x = BLOCK_LENGTH;
        if ( taskid > (numworkers - ROOT_WORKERS) ) {
            end_x = BLOCK_LENGTH-1;
            down = NONE;
        } 
	
	left = taskid - 1;
	start_y = 1;
        if ( taskid%(ROOT_WORKERS) == 1) {
            start_y = 2;
            left= NONE;
        } 
	
	right = taskid + 1; 
	end_y = BLOCK_LENGTH;
        if ( taskid%(ROOT_WORKERS) == 0 ) {
            end_y = BLOCK_LENGTH-1;
            right = NONE;
        }         
        
        /* Worker to worker datatypes */
        MPI_Datatype column;
        MPI_Type_vector ( BLOCK_LENGTH, 1, BLOCK_LENGTH+2, MPI_FLOAT, &column );
        MPI_Type_commit ( &column );

	/* All send and receive requests */
        MPI_Request req_s_left,req_s_right,req_s_top,req_s_down;
        MPI_Request req_r_left,req_r_right,req_r_top,req_r_down;

        /* Begin doing STEPS iterations.  Must communicate border rows with */
        /* neighbors.  If I have the first or last grid row, then I only need */
        /*  to  communicate with one neighbor  */
        printf ( "Task %d received work. Beginning time steps...\n",taskid );
        
	
	
	MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();          
        iz = 0;
        
       #pragma omp parallel num_threads(THREADS) private( it )
       {                   
        for ( it = 1; it <= STEPS; it++ ) {
		#pragma omp master
		{
            if ( left != NONE ) {                
                MPI_Isend ( &u[iz][1][1], 1, column, left, RTAG, MPI_COMM_WORLD,&req_s_left );
                source = left;
                msgtype = LTAG;

                MPI_Irecv ( &u[iz][1][0], 1, column, source, msgtype, MPI_COMM_WORLD,&req_r_left );

            }
            
            if ( top != NONE ) {
                MPI_Isend ( &u[iz][1][1], BLOCK_LENGTH, MPI_FLOAT, top , DTAG, MPI_COMM_WORLD,&req_s_top );
         	source = top;
                msgtype = TTAG;

                MPI_Irecv ( &u[iz][0][1], BLOCK_LENGTH, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD,&req_r_top );

            }
            
            if ( right != NONE ) {
                MPI_Isend ( &u[iz][1][BLOCK_LENGTH], 1,column , right , LTAG, MPI_COMM_WORLD,&req_s_right );         
                source = right;
                msgtype = RTAG;

                MPI_Irecv ( &u[iz][1][BLOCK_LENGTH+1], 1, column , source, msgtype, MPI_COMM_WORLD, &req_r_right );

            }
            
            if ( down != NONE ) {
                MPI_Isend ( &u[iz][BLOCK_LENGTH][1], BLOCK_LENGTH, MPI_FLOAT , down, TTAG, MPI_COMM_WORLD,&req_s_down );
                source = down;
                msgtype = DTAG;

                MPI_Irecv ( &u[iz][BLOCK_LENGTH+1][1], BLOCK_LENGTH, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD,&req_r_down );

            }        
            }
            /* Now call update to update the value of inner grid points */
            #pragma omp barrier
            update ( start_x+1,end_x-1,start_y+1,end_y-1,BLOCK_LENGTH+2,&u[iz][0][0],&u[1-iz][0][0],1,taskid );            
                        
            #pragma omp master
		{
            if ( left != NONE ) {
                MPI_Wait ( &req_r_left, MPI_STATUS_IGNORE );
            }
            if ( right != NONE ) {
                MPI_Wait ( &req_r_right, MPI_STATUS_IGNORE );
            }
            if ( top != NONE ) {
                MPI_Wait ( &req_r_top, MPI_STATUS_IGNORE );
            }
            if ( down != NONE ) {
                MPI_Wait ( &req_r_down, MPI_STATUS_IGNORE );
            }
            }
            /* Now call update to update the value of grid points across the borders */
            #pragma omp barrier
            update ( start_x+1,end_x-1,start_y,start_y,BLOCK_LENGTH+2,&u[iz][0][0],&u[1-iz][0][0],0,taskid );
            update ( start_x+1,end_x-1,end_y,end_y,BLOCK_LENGTH+2,&u[iz][0][0],&u[1-iz][0][0] ,0,taskid);
            update ( start_x,start_x,start_y,end_y,BLOCK_LENGTH+2,&u[iz][0][0],&u[1-iz][0][0],0,taskid );
            update ( end_x,end_x,start_y,end_y,BLOCK_LENGTH+2,&u[iz][0][0],&u[1-iz][0][0],0,taskid );
                        
            #pragma omp master
		{
            if ( left != NONE ) {
                MPI_Wait ( &req_s_left, MPI_STATUS_IGNORE );
            }
            if ( right != NONE ) {
                MPI_Wait ( &req_s_right, MPI_STATUS_IGNORE );
            }
            if ( top != NONE ) {
                MPI_Wait ( &req_s_top, MPI_STATUS_IGNORE );
            }
            if ( down != NONE ) {
                MPI_Wait ( &req_s_down, MPI_STATUS_IGNORE );
            }                 

	    /* DEBUGGING - print per taskid
	    
	    if (taskid==1 && it==STEPS){
		    for ( iy = BLOCK_LENGTH; iy >= 1; iy-- ) {
			for ( ix = 1; ix <= BLOCK_LENGTH; ix++ ) {
			    printf ( "%6.1f", * ( &u[0][0][0]+ix*(BLOCK_LENGTH+2)+iy ) );
			    if ( ix != BLOCK_LENGTH ) {
				printf (" " );
			    } else {
				printf ("\n" );
			    }
			}
	    		}
    		}*/
    	
    		
	#if CONVERGENCE_CHK
    	    /* Convergence check */    
    	    if (it%INTERVAL == 0){    
	    	    int exit = 1;
	    	    for (ix=1;ix<BLOCK_LENGTH+1;ix++){
	    	    	for (iy=1;iy<BLOCK_LENGTH+1;iy++){		    	    		
	    	    		if ( ( u[iz][ix][iy] - u[1-iz][ix][iy] ) >= SENSITIVITY ){
	    	    			exit = 0;	    	    			
	    	    			break;
	    	    		}
	    	    	}
	    	    	if (!exit){	    	    	
	    	    		break;
	    	    	}
	    	    }
	    	    if (exit){	    	        
	    	    	break;
	    	    }
	    }
	#endif
	
            iz = 1 - iz;
            
            
             }
             
        }    }
        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        local_elapsed = end_time - start_time;
        MPI_Type_free ( &column );      
        if (taskid == 1) 
        	printf ( ">> Tasks finished their work at : %d iterations!\n",( (it>STEPS) ? STEPS : it) );	
    }
    /* Find max elapsed time of all processes by using MPI_Reduce */
    MPI_Reduce ( &local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD );
    if ( taskid == MASTER ) {
        printf ( "Time = %.3f seconds\n",elapsed );
    }
    MPI_Finalize();
}


/**************************************************************************
*  subroutine update
****************************************************************************/
void update ( int start_x, int end_x,int start_y, int end_y,int ny, float *u1, float *u2,int mode, int taskid )
{ 

    int ix, iy;    

    #pragma omp for schedule(static) collapse(2)    
    for ( ix = start_x; ix <= end_x; ix++ ){        	
        for ( iy = start_y; iy <= end_y; iy++ )   {     
            * ( u2+ix*ny+iy ) = * ( u1+ix*ny+iy )  +
                                parms.cx * ( * ( u1+ ( ix+1 ) *ny+iy ) +
                                             * ( u1+ ( ix-1 ) *ny+iy ) -
                                             2.0 * * ( u1+ix*ny+iy ) ) +
                                parms.cy * ( * ( u1+ix*ny+iy+1 ) +
                                             * ( u1+ix*ny+iy-1 ) -
                                             2.0 * * ( u1+ix*ny+iy ) );
                                             
             //if (taskid==4 && mode==1)printf("DTGTGGGTT Thread: %d ix=%d iy=%d \n",omp_get_thread_num(),ix,iy);


	}
	
     }
     
}

/*****************************************************************************
*  subroutine inidat
*****************************************************************************/
void inidat ( int nx, int ny, float *u, int taskid, int no_workers_one_dimension )
{  
    int ix, iy;
    int offset_x = ((taskid-1)/no_workers_one_dimension)*BLOCK_LENGTH;
    int offset_y = ((taskid-1)%no_workers_one_dimension)*BLOCK_LENGTH;

    for ( ix = 1; ix <= nx-2; ix++ ){
        for ( iy = 1; iy <= ny-2; iy++ ) {       
        
            * ( u+ix*ny+iy ) = ( float ) ( (ix+offset_x-1) * ( BLOCK_LENGTH*no_workers_one_dimension - (ix+offset_x-1) - 1 ) 
            							* (iy+offset_y-1) * ( BLOCK_LENGTH*no_workers_one_dimension 
            							- (iy+offset_y-1) - 1 ) );
                            		
        }        
     }
}

/**************************************************************************
* subroutine prtdat
**************************************************************************/
void prtdat ( int nx, int ny, float *u1, char *fnam )
{
    int ix, iy;
    FILE *fp;

    fp = fopen ( fnam, "w" );
    for ( iy = ny-2; iy >= 1; iy-- ) {
        for ( ix = 1; ix <= nx-2; ix++ ) {
            fprintf ( fp, "%6.1f", * ( u1+ix*ny+iy ) );
            if ( ix != nx-2 ) {
                fprintf ( fp, " " );
            } else {
                fprintf ( fp, "\n" );
            }
        }
    }
    fclose ( fp );
}

/**************************************************************************
**************************************************************************/

