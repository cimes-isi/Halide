#include <mpi.h>

#include "DistributeLoops.h"
#include "Function.h"
#include "IRMutator.h"
#include "IROperator.h"

namespace Halide {
namespace Internal {

// For each distributed for loop, mutate its bounds to be determined
// by processor rank.
class DistributeLoops : public IRMutator {
public:
    using IRMutator::visit;
    Stmt visit(const For *op) {
        if (op->distributed) {
            Expr rank = Call::make(Int(32), Call::mpi_rank, {}, Call::PureIntrinsic);
            Expr num_processors = Call::make(Int(32), Call::mpi_num_processors, {}, Call::PureIntrinsic);
            // Number of elements each processor needs to process.
            Expr slice_size = (op->extent + num_processors - 1) / num_processors;
            Expr new_min = op->min + slice_size * rank;
            Expr new_max = new_min + slice_size - 1;
            Expr new_extent = min(new_max, op->min + op->extent - 1) - new_min + 1;
            return For::make(op->name, new_min, new_extent, op->for_type, op->distributed, op->device_api, mutate(op->body));
        } else {
            return IRMutator::visit(op);
        }
    }
};

Stmt distribute_loops(Stmt s) {
    s = DistributeLoops().mutate(s);
    return s;
}

} // Internal
} // Halide