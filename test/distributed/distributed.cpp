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

    { // same test as above, but with a non-divisible extent
        Func f;
        f(x) = 2 * x + 1;
        f.distribute(x);
        Buffer<int> out = f.realize(25);
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
        int num_elements = 23;
        f.infer_input_bounds({num_elements}, get_jit_target_from_environment(), { { in, &in_buf } });
        int num_elements_per_proc = (num_elements + numprocs - 1) / numprocs;
        int buf_min = num_elements_per_proc * rank;
        int buf_max = std::min(buf_min + num_elements_per_proc - 1, num_elements - 1);
        if (in_buf.dim(0).min() != buf_min) {
            printf("rank %d: in_buf.dim(0).min() = %d instead of %d\n", rank, in_buf.dim(0).min(), buf_min);
            MPI_Finalize();
            return -1;
        }
        if (in_buf.dim(0).max() != buf_max) {
            printf("rank %d: in_buf.dim(0).max() = %d instead of %d\n", rank, in_buf.dim(0).max(), buf_max);
            MPI_Finalize();
            return -1;
        }
        for (int x = in_buf.dim(0).min(); x <= in_buf.dim(0).max(); x++) {
            in_buf(x) = 2 * x;
        }
        in.set(in_buf);

        Buffer<int> out = f.realize(num_elements);
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
