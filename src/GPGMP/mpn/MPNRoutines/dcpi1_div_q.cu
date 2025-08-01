/* gpmpn_dc_div_q -- divide-and-conquer division, returning exact quotient
   only.

   Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2007, 2009, 2010 Free Software Foundation, Inc.

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

namespace gpgmp {
  namespace mpnRoutines {

    ANYCALLER mp_size_t gpmpn_dcpi1_div_q_itch(mp_size_t nn, mp_size_t dn)
    {
      mp_size_t totalScratchNeeded = 0;
      mp_size_t qn = nn - dn;

      totalScratchNeeded += (nn + 1);

      totalScratchNeeded += (qn + 1);

      return totalScratchNeeded;
    }

    ANYCALLER mp_size_t gpmpn_dcpi1_div_q_itch_maximum(mp_size_t maximumLimbCountBetweenNumeratorAndDenominator)
    {
      return gpmpn_dcpi1_div_q_itch(maximumLimbCountBetweenNumeratorAndDenominator, 1);
    }

    HOSTONLY mp_limb_t gpmpn_dcpi1_div_q(mp_ptr qp, mp_ptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn, gmp_pi1_t *dinv, mp_limb_t* scratchSpace)
    {
      mp_ptr tp, wp;
      mp_limb_t qh;
      mp_size_t qn;

      ASSERT (dn >= 6);
      ASSERT (nn - dn >= 3);
      ASSERT (dp[dn-1] & GMP_NUMB_HIGHBIT);

      tp = scratchSpace;
      scratchSpace += (nn + 1);
      MPN_COPY (tp + 1, np, nn);
      tp[0] = 0;

      qn = nn - dn;
      wp = scratchSpace;
      scratchSpace += (qn + 1);

      qh = gpmpn_dcpi1_divappr_q (wp, tp, nn + 1, dp, dn, dinv);

      if (wp[0] == 0)
      {
        mp_limb_t cy;

        if (qn > dn)
        {
          gpmpn_mul (tp, wp + 1, qn, dp, dn);
        }
        else
        {
          gpmpn_mul (tp, dp, dn, wp + 1, qn);
        }

        cy = (qh != 0) ? gpmpn_add_n (tp + qn, tp + qn, dp, dn) : 0;

        if (cy || gpmpn_cmp (tp, np, nn) > 0) /* At most is wrong by one, no cycle. */
        {
          qh -= gpmpn_sub_1 (qp, wp + 1, qn, 1);
        }
        else /* Same as below */
        {
          MPN_COPY (qp, wp + 1, qn);
        }
      }
      else
      {
        MPN_COPY (qp, wp + 1, qn);
      }

      return qh;
    }

  }
}