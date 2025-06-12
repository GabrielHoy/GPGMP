/* gpmpn_mul_n -- multiply natural numbers.

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

    //This function has been significantly trimmed down and all TOOM multiplication removed for the sake of minimizing Warp Divergence.
    //It is likely that this won't be as fast as GMP due to this.
    ANYCALLER void gpmpn_mul_n(mp_ptr p, mp_srcptr a, mp_srcptr b, mp_size_t n)
    {
      ASSERT(n >= 1);
      ASSERT(!MPN_OVERLAP_P(p, 2 * n, a, n));
      ASSERT(!MPN_OVERLAP_P(p, 2 * n, b, n));

      if (BELOW_THRESHOLD(n, MUL_FFT_THRESHOLD))
      {
        gpmpn_mul_basecase(p, a, n, b, n);
      }
      else
      {
        /* The current FFT code allocates its own space.  That should probably
     change.  */
        gpmpn_fft_mul(p, a, n, b, n);
      }
    }

  }
}