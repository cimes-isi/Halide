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

    {
        ImageParam in(Int(32), 1);
        Func f;
        f(x) = 2 * in(x) + 1;
        f.distribute(x);
        Buffer<int> in_buf;
        f.infer_input_bounds({20}, get_jit_target_from_environment(), { { in, &in_buf } });
        if (in_buf.dim(0).min() != rank * 10) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), rank * 10);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != rank * 10 + 9) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), rank * 10 + 9);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }
        in.set(in_buf);

        Buffer<int> out = f.realize(20);
        for (int x = out.dim(0).min(); x <= out.dim(0).max(); x++) {
            int correct = 4 * x + 1;
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
