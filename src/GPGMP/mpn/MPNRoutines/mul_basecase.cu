/* gpmpn_mul_basecase -- Internal routine to multiply two natural numbers
   of length m and n.

   THIS IS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH THIS FUNCTION THROUGH DOCUMENTED INTERFACES.

Copyright 1991-1994, 1996, 1997, 2000-2002 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "GPGMP/gpgmp-impl.cuh"
#include "GPGMP/longlong.cuh"

namespace gpgmp
{
  namespace mpnRoutines
  {

    /* Multiply {up,usize} by {vp,vsize} and write the result to
       {prodp,usize+vsize}.  Must have usize>=vsize.

       Note that prodp gets usize+vsize limbs stored, even if the actual result
       only needs usize+vsize-1.

       There's no good reason to call here with vsize>=MUL_TOOM22_THRESHOLD.
       Currently this is allowed, but it might not be in the future.

       This is the most critical code for multiplication.  All multiplies rely
       on this, both small and huge.  Small ones arrive here immediately, huge
       ones arrive here as this is the base case for Karatsuba's recursive
       algorithm.  */

    ANYCALLER void gpmpn_mul_basecase(mp_ptr rp, mp_srcptr up, mp_size_t un, mp_srcptr vp, mp_size_t vn)
    {
      ASSERT(un >= vn);
      ASSERT(vn >= 1);
      ASSERT(!MPN_OVERLAP_P(rp, un + vn, up, un));
      ASSERT(!MPN_OVERLAP_P(rp, un + vn, vp, vn));

      /* We first multiply by the low order limb (or depending on optional function
         availability, limbs).  This result can be stored, not added, to rp.  We
         also avoid a loop for zeroing this way.  */

      rp[un] = gpmpn_mul_1(rp, up, un, vp[0]);
      rp += 1, vp += 1, vn -= 1;

      /* Now accumulate the product of up[] and the next higher limb (or depending
         on optional function availability, limbs) from vp[].  */

#define MAX_LEFT MP_SIZE_T_MAX /* Used to simplify loops into if statements */

      while (vn >= 1)
      {
        rp[un] = gpmpn_addmul_1(rp, up, un, vp[0]);
        //I do not understand why this check was here originally; with the decade+ of development GMP has I'm sure there's something I'm missing but after a little bit of searching it seems like this would never actually trigger a return....
        //I've removed it for now for the sake of speed since this is very much a hotpath.
        //if (MAX_LEFT == 1)
        //  return;
        rp += 1, vp += 1, vn -= 1;
      }
    }

  }
}