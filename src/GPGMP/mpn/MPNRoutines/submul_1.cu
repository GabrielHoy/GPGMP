/* gpmpn_submul_1 -- multiply the N long limb vector pointed to by UP by VL,
   subtract the N least significant limbs of the product from the limb
   vector pointed to by RP.  Return the most significant limb of the
   product, adjusted for carry-out from the subtraction.

Copyright 1992-1994, 1996, 2000, 2002, 2004 Free Software Foundation, Inc.

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

    ANYCALLER mp_limb_t gpmpn_submul_1(mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t v0)
    {
      mp_limb_t u0, crec, c, p1, p0, r0;

      ASSERT(n >= 1);
      ASSERT(MPN_SAME_OR_SEPARATE_P(rp, up, n));

      crec = 0;
      do
      {
        u0 = *up++;
        umul_ppmm(p1, p0, u0, v0);

        r0 = *rp;

        p0 = r0 - p0;
        c = r0 < p0;

        p1 = p1 + c;

        r0 = p0 - crec; /* cycle 0, 3, ... */
        c = p0 < r0;    /* cycle 1, 4, ... */

        crec = p1 + c; /* cycle 2, 5, ... */

        *rp++ = r0;
      } while (--n != 0);

      return crec;
    }

  }
}