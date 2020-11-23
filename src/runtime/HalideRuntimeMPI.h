#ifndef HALIDE_HALIDERUNTIMEMPI_H
#define HALIDE_HALIDERUNTIMEMPI_H

// Don't include HalideRuntime.h if the contents of it were already pasted into a generated header above this one
#ifndef HALIDE_HALIDERUNTIME_H

#include "HalideRuntime.h"

#endif

#ifdef __cplusplus
extern "C" {
#endif

/** \file
 *  Routines specific to the Halide MPI runtime.
 */

extern int halide_mpi_num_processors();
extern int halide_mpi_rank();

#ifdef __cplusplus
}  // End extern "C"
#endif

#endif  // HALIDE_HALIDERUNTIMEMPI_H
