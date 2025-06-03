//Just includes all relevant MPN files
#pragma once
//General MPN functions used by GPGMP...
#include "PredictSizes.cuh"
#include "Allocation.cuh"
#include "Init.cuh"
#include "CopyArrayToArray.cuh"
#include "CopyToMPZArray.cuh"

//"MPN" arithmetic routines used by GPGMP, most-to-all of these are ported over from the GMP library.
#include "MPNRoutines/add_1.cuh"
//#include "MPNRoutines/add_err1_n.cuh"
//#include "MPNRoutines/add_err2_n.cuh"
//#include "MPNRoutines/add_err3_n.cuh"
//#include "MPNRoutines/add_n_sub_n.cuh"
#include "MPNRoutines/add_n.cuh"

#include "MPNRoutines/zero.cuh"