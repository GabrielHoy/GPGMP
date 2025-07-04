/* mpf_pow_ui -- Compute b^e.

Copyright 1998, 1999, 2001, 2012, 2015 Free Software Foundation, Inc.

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
  namespace mpfRoutines
  {

    /* This uses a plain right-to-left square-and-multiply algorithm.

       FIXME: When popcount(e) is not too small, it would probably speed things up
       to use a k-ary sliding window algorithm.  */

    ANYCALLER void
    gpmpf_pow_ui(mpf_ptr r, mpf_srcptr b, unsigned long int e, mp_limb_t* scratchSpace)
    {
      mpf_t t;
      int cnt;

      if (e <= 1)
      {
        if (e == 0)
          gpmpf_set_ui(r, 1);
        else
          gpmpf_set(r, b);
        return;
      }

      count_leading_zeros(cnt, (mp_limb_t)e);
      cnt = GMP_LIMB_BITS - 1 - cnt;

      /* Increase computation precision as a function of the exponent.  Adding
         log2(popcount(e) + log2(e)) bits should be sufficient, but we add log2(e),
         i.e. much more.  With mpf's rounding of precision to whole limbs, this
         will be excessive only when limbs are artificially small.  */
      gpmpf_init2(t, gpmpf_get_prec(r) + cnt);

      gpmpf_set(t, b); /* consume most significant bit */
      while (--cnt > 0)
      {
        gpmpf_mul(t, t, t, scratchSpace);
        if ((e >> cnt) & 1)
          gpmpf_mul(t, t, b, scratchSpace);
      }

      /* Do the last iteration specially in order to save a copy operation.  */
      if (e & 1)
      {
        gpmpf_mul(t, t, t, scratchSpace);
        gpmpf_mul(r, t, b, scratchSpace);
      }
      else
      {
        gpmpf_mul(r, t, t, scratchSpace);
      }

      gpmpf_clear(t);
    }

  }
}