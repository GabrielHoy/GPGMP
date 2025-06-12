/* gpmpn_sqr -- square natural numbers.

Copyright 1991, 1993, 1994, 1996-2003, 2005, 2008, 2009 Free Software
Foundation, Inc.

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

#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpnRoutines
  {

    //This function has been significantly trimmed down due to issues it poses on the GPU.
    //It will not be as fast as the original GMP's implementation in most cases; all TOOM paths are gone.
    ANYCALLER void gpmpn_sqr(mp_ptr p, mp_srcptr a, mp_size_t n)
    {
      ASSERT(n >= 1);
      ASSERT(!MPN_OVERLAP_P(p, 2 * n, a, n));

      //Realistically this check introduces possible warp divergence and for the GPU I feel is less than necessary -- gpmpn_fft_mul furthers that divergence as well.
      //Let's just use the basecase.
      gpmpn_sqr_basecase(p, a, n);

      //If anyone is taking a look at this code; this is what I had previously after removing TOOM paths - will preserve for reference
      /* if (BELOW_THRESHOLD(n, SQR_BASECASE_THRESHOLD))
      { // mul_basecase is faster than sqr_basecase on small sizes sometimes
        gpmpn_mul_basecase(p, a, n, a, n);
      }
      else if (BELOW_THRESHOLD(n, SQR_FFT_THRESHOLD))
      {
        gpmpn_sqr_basecase(p, a, n);
      }
      else
      {
        //havent tested this path so i sure hope it works :^)
        // The current FFT code allocates its own space.  That should probably
        // change.
         gpmpn_fft_mul(p, a, n, a, n);
      } */
    }

  }
}