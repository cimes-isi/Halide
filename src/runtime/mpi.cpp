#include "runtime_internal.h"
#include "HalideRuntimeMPI.h"
#include <mpi.h>

extern "C" {

WEAK int halide_mpi_num_processors() {
	int num_processes = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	return num_processes;
}

WEAK int halide_mpi_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

} // extern "C"
