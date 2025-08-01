/* mpf_reldiff -- Generate the relative difference of two floats.

Copyright 1996, 2001, 2004, 2005 Free Software Foundation, Inc.

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

namespace gpgmp
{
  namespace mpfRoutines
  {

    /* The precision we use for d = x-y is based on what mpf_div will want from
       the dividend.  It calls mpn_div_q to produce a quotient of rprec+1 limbs.
       So rprec+1 == dsize - xsize + 1, hence dprec = rprec+xsize.  */

    ANYCALLER void
    gpmpf_reldiff(mpf_ptr rdiff, mpf_srcptr x, mpf_srcptr y, mp_limb_t* scratchSpace)
    {
      if (UNLIKELY(SIZ(x) == 0))
      {
        gpmpf_set_ui(rdiff, (unsigned long int)(mpf_sgn(y) != 0));
      }
      else
      {
        mp_size_t dprec;
        mpf_t d;
        TMP_DECL;

        TMP_MARK;
        dprec = PREC(rdiff) + ABSIZ(x);
        ASSERT(PREC(rdiff) + 1 == dprec - ABSIZ(x) + 1);

        PREC(d) = dprec;
        PTR(d) = scratchSpace;

        gpmpf_sub(d, x, y, scratchSpace);
        SIZ(d) = ABSIZ(d);
        gpmpf_div(rdiff, d, x, scratchSpace);

        TMP_FREE;
      }
    }

  }
}