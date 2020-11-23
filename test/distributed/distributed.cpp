#include "Halide.h"
#include <mpi.h>
#include <iomanip>
#include <stdarg.h>
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    Var x, y, z, c;

    {
        Func f;
        f(x) = 2 * x + 1;
        f.distribute(x);
        Buffer<int> out = f.realize(20);
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 2 * x + 1;
            if (out(x) != correct) {
                printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
