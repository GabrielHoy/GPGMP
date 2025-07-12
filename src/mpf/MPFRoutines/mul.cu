/* mpf_mul -- Multiply two floats.

Copyright 1993, 1994, 1996, 2001, 2005, 2019, 2020 Free Software
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

namespace gpgmp
{
  namespace mpfRoutines
  {

    ANYCALLER void gpmpf_mul(mpf_ptr r, mpf_srcptr u, mpf_srcptr v, mp_limb_t* scratchSpace)
    {
      mp_size_t sign_product;
      mp_size_t prec = PREC(r);
      mp_size_t rsize;
      mp_limb_t cy_limb;
      mp_ptr rp;
      mp_size_t adj;

      if (u == v)
      {
        mp_srcptr up;
        mp_size_t usize;

        sign_product = 0;

        usize = ABSIZ(u);

        up = PTR(u);
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }

        if (usize == 0)
        {
          SIZ(r) = 0;
          EXP(r) = 0; /* ??? */
          return;
        }
        else
        {
          rsize = 2 * usize;

          gpgmp::mpnRoutines::gpmpn_sqr(scratchSpace, up, usize);
          cy_limb = scratchSpace[rsize - 1];
        }
      }
      else
      {
        mp_srcptr up, vp;
        mp_size_t usize, vsize;

        usize = SIZ(u);
        vsize = SIZ(v);
        sign_product = usize ^ vsize;

        usize = ABS(usize);
        vsize = ABS(vsize);

        up = PTR(u);
        vp = PTR(v);
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }
        if (vsize > prec)
        {
          vp += vsize - prec;
          vsize = prec;
        }

        if (usize == 0 || vsize == 0)
        {
          SIZ(r) = 0;
          EXP(r) = 0;
          return;
        }
        else
        {
          rsize = usize + vsize;
          cy_limb = (usize >= vsize
                         ? gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, up, usize, vp, vsize)
                         : gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, vp, vsize, up, usize));
        }
      }

      adj = cy_limb == 0;
      rsize -= adj;
      prec++;
      if (rsize > prec)
      {
        scratchSpace += rsize - prec;
        rsize = prec;
      }
      rp = PTR(r);
      MPN_COPY(rp, scratchSpace, rsize);
      EXP(r) = EXP(u) + EXP(v) - adj;
      SIZ(r) = sign_product >= 0 ? rsize : -rsize;
    }

  }
}