/* gpmpn_add_err2_n -- add_n with two error terms

   Contributed by David Harvey.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2011 Free Software Foundation, Inc.

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

#pragma once
#include "gpgmp-impl.cuh"

namespace gpgmp {

	namespace mpnRoutines {

    /*
      Computes:

      (1) {rp,n} := {up,n} + {vp,n} (just like gpmpn_add_n) with incoming carry cy,
      return value is carry out.

      (2) Let c[i+1] = carry from i-th limb addition (c[0] = cy).
      Computes c[1]*yp1[n-1] + ... + c[n]*yp1[0],
              c[1]*yp2[n-1] + ... + c[n]*yp2[0],
      stores two-limb results at {ep,2} and {ep+2,2} respectively.

      Requires n >= 1.

      None of the outputs may overlap each other or any of the inputs, except
      that {rp,n} may be equal to {up,n} or {vp,n}.
    */
    ANYCALLER mp_limb_t gpmpn_add_err2_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_ptr ep, mp_srcptr yp1, mp_srcptr yp2, mp_size_t n, mp_limb_t cy)
    {
      mp_limb_t el1, eh1, el2, eh2, ul, vl, yl1, yl2, zl1, zl2, rl, sl, cy1, cy2;

      ASSERT (n >= 1);
      ASSERT (MPN_SAME_OR_SEPARATE_P (rp, up, n));
      ASSERT (MPN_SAME_OR_SEPARATE_P (rp, vp, n));
      ASSERT (! MPN_OVERLAP_P (rp, n, yp1, n));
      ASSERT (! MPN_OVERLAP_P (rp, n, yp2, n));
      ASSERT (! MPN_OVERLAP_P (ep, 4, up, n));
      ASSERT (! MPN_OVERLAP_P (ep, 4, vp, n));
      ASSERT (! MPN_OVERLAP_P (ep, 4, yp1, n));
      ASSERT (! MPN_OVERLAP_P (ep, 4, yp2, n));
      ASSERT (! MPN_OVERLAP_P (ep, 4, rp, n));

      yp1 += n - 1;
      yp2 += n - 1;
      el1 = eh1 = 0;
      el2 = eh2 = 0;

      do
        {
          yl1 = *yp1--;
          yl2 = *yp2--;
          ul = *up++;
          vl = *vp++;

          /* ordinary add_n */
          ADDC_LIMB (cy1, sl, ul, vl);
          ADDC_LIMB (cy2, rl, sl, cy);
          cy = cy1 | cy2;
          *rp++ = rl;

          /* update (eh1:el1) */
          zl1 = (-cy) & yl1;
          el1 += zl1;
          eh1 += el1 < zl1;

          /* update (eh2:el2) */
          zl2 = (-cy) & yl2;
          el2 += zl2;
          eh2 += el2 < zl2;
        }
      while (--n);

      ep[0] = el1;
      ep[1] = eh1;
      ep[2] = el2;
      ep[3] = eh2;

      return cy;
    }

  }
}
