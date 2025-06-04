//Just includes all relevant MPN files
#pragma once
//General MPN functions used by GPGMP...
#include "PredictSizes.cuh"
#include "Allocation.cuh"
#include "Init.cuh"
#include "CopyArrayToArray.cuh"
#include "CopyToMPZArray.cuh"

//"MPN" routines used by GPGMP.
//Most-to-all of these are ported over from the GMP library, and possibly rewritten.
//https://gmplib.org/manual/Low_002dlevel-Functions
#include "MPNRoutines/add_1.cuh"
#include "MPNRoutines/add_err1_n.cuh"
#include "MPNRoutines/add_err2_n.cuh"
#include "MPNRoutines/add_err3_n.cuh"
#include "MPNRoutines/add_n_sub_n.cuh" //Warp Divergence
#include "MPNRoutines/add_n.cuh"
#include "MPNRoutines/add.cuh"
#include "MPNRoutines/com.cuh"
#include "MPNRoutines/copyd.cuh"
#include "MPNRoutines/copyi.cuh"
#include "MPNRoutines/zero_p.cuh"
#include "MPNRoutines/zero.cuh"
#include "MPNRoutines/cnd_swap.cuh"
#include "MPNRoutines/cmp.cuh"
#include "MPNRoutines/cnd_add_n.cuh"
#include "MPNRoutines/neg.cuh"
#include "MPNRoutines/sub.cuh"
#include "MPNRoutines/sub_n.cuh"
#include "MPNRoutines/sub_1.cuh"
#include "MPNRoutines/sub_err1_n.cuh"
#include "MPNRoutines/sub_err2_n.cuh"
#include "MPNRoutines/sub_err3_n.cuh"
#include "MPNRoutines/cnd_sub_n.cuh"
#include "MPNRoutines/addmul_1.cuh"
//#include "MPNRoutines/bsqrt.cuh" //Lots of Sub-Routines
//#include "MPNRoutines/bsqrtinv.cuh" //Warp Divergence + Lots of Sub-Routines
#include "MPNRoutines/rshift.cuh"
#include "MPNRoutines/lshift.cuh"
#include "MPNRoutines/lshiftc.cuh"
#include "MPNRoutines/dump.cuh"
