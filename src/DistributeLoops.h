#ifndef HALIDE_DISTRIBUTE_LOOPS_H
#define HALIDE_DISTRIBUTE_LOOPS_H

/** \file
 * Defines the lowering pass that distributes loops marked as such
 */

#include "Bounds.h"
#include "IR.h"

#include <map>

namespace Halide {
namespace Internal {

/** Take a statement with for loops marked for distribution, and turn
 * them into loops that operate on a subset of their input data
 * according to their MPI rank. Return true if it finds a distributed loop.
 */
std::pair<Stmt, bool> distribute_loops(Stmt s);

} // namespace Internal
} // namespace Halide

#endif
