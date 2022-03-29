#define LAB4_EXTEND

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "Lab4_IO.h"
#include "timer.h"

#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85

#define THRESHOLD 0.0001

/*
* Algorithm:
* - Copy local r to local r pre
* - Calculate new r
* - Gather all r to process 0
* - Process 0 calculates contributeion
* - Process 0 checks finished condition
* - Not finished, broadcast new contributions
*/

int main (int argc, char* argv[]){
    struct node *nodehead;
    int nodecount;
    int local_nodecount;
    double *r, *r_pre, *contribution;
    double *local_r, *local_r_pre, *local_contribution;
    int i, j;
    double damp_const;
    double error;
    FILE *fp, *ip;
    int node_diff = 0;

    int done = 0;

    int my_rank, comm_sz;
    MPI_Comm comm;

    double start;
    double end;
    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    // Get the input data
    if ((ip = fopen("data_input_meta","r")) == NULL) {
        printf("Error opening the data_input_meta file.\n");
        return 254;
    }
    fscanf(ip, "%d\n", &nodecount);
    fclose(ip);

    local_nodecount = nodecount/comm_sz;
    if ((local_nodecount * comm_sz) != nodecount) {
        local_nodecount++;
    }
    
    if (node_init(&nodehead, 0, nodecount)) return 254;
    // initialize variables
    if (my_rank == 0) {
        r = malloc(nodecount * sizeof(double));
        r_pre = malloc(nodecount * sizeof(double));
    }
    local_r = malloc(local_nodecount * sizeof(double));
    local_r_pre = malloc(local_nodecount * sizeof(double));
    local_contribution = malloc(local_nodecount * sizeof(double));
    for ( i = 0; i < local_nodecount; ++i) 
        local_r[i] = 1.0 / nodecount;
    if (my_rank == comm_sz-1) {
        node_diff = (local_nodecount * comm_sz) - nodecount;
        for ( i = 0; i < node_diff; ++i) {
            int index = local_nodecount - 1;
            index -= i;
            local_r[index] = 0;
        }
    }
    contribution = malloc(nodecount * sizeof(double));
    if (my_rank == 0) {
        for ( i = 0; i < nodecount; ++i) {
            contribution[i] = r[i] / nodehead[i].num_out_links * DAMPING_FACTOR;
        }
    }
    MPI_Barrier(comm);
    MPI_Bcast(contribution, nodecount, MPI_DOUBLE, 0, comm);
    damp_const = (1.0 - DAMPING_FACTOR) / nodecount;

    // CORE CALCULATION
    MPI_Barrier(comm);
    GET_TIME(start);
    while(done != 1){
        vec_cp(local_r, local_r_pre, local_nodecount);
        int offset = local_nodecount*my_rank;
        // update the local r values
        for ( i = 0; i < local_nodecount - node_diff; ++i){
            local_r[i] = 0;
            for ( j = 0; j < nodehead[i+offset].num_in_links; ++j)
                local_r[i] += contribution[nodehead[i+offset].inlinks[j]];
            local_r[i] += damp_const;
        }
        // update and broadcast the contribution
        for ( i=0; i<local_nodecount - node_diff; ++i){
            local_contribution[i] = local_r[i] / nodehead[i+offset].num_out_links * DAMPING_FACTOR;
        }
        MPI_Allgather(local_contribution, local_nodecount-node_diff, MPI_DOUBLE,
                    contribution, local_nodecount-node_diff, MPI_DOUBLE, comm);
        MPI_Gather(local_r, local_nodecount-node_diff, MPI_DOUBLE,
                    r, local_nodecount-node_diff, MPI_DOUBLE, 0, comm);
        // Rank 0 checks for error and broadcasts if converged
        if (my_rank == 0) {
            if (rel_error(r, r_pre, nodecount) < EPSILON) {
                done = 1;
            } else {
                done = 0;
                vec_cp(r, r_pre, nodecount);
            }
        }
        MPI_Bcast(&done, 1, MPI_INT, 0, comm);
    }
    GET_TIME(end);

    // Save the result
    if (my_rank ==0) {
        Lab4_saveoutput(r, nodecount, end-start);
    }

    // post processing
    MPI_Barrier(comm);
    node_destroy(nodehead, nodecount);
    free(contribution);
    free(local_r);
    free(r_pre);
    free(local_r_pre);
    free(local_contribution);
    if (my_rank == 0) {
        free(r);
    }

    MPI_Finalize();
    return 0;
}
