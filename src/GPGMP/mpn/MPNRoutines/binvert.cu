/* Compute {up,n}^(-1) mod B^n.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright (C) 2004-2007, 2009, 2012, 2017, 2021 Free Software Foundation, Inc.

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

    /*
      r[k+1] = r[k] - r[k] * (u*r[k] - 1)
      r[k+1] = r[k] + r[k] - r[k]*(u*r[k])
    */

    #if TUNE_PROGRAM_BUILD
    #define NPOWS \
    ((sizeof(mp_size_t) > 6 ? 48 : 8*sizeof(mp_size_t)))
    #else
    #define NPOWS \
    ((sizeof(mp_size_t) > 6 ? 48 : 8*sizeof(mp_size_t)) - LOG2C (BINV_NEWTON_THRESHOLD))
    #endif

    ANYCALLER mp_size_t gpmpn_binvert_itch (mp_size_t n)
    {
      mp_size_t itch_local = gpmpn_mulmod_bnm1_next_size (n);
      mp_size_t itch_out = gpmpn_mulmod_bnm1_itch (itch_local, n, (n + 1) >> 1);
      return itch_local + itch_out;
    }

    ANYCALLER void gpmpn_binvert (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_ptr scratch)
    {
      mp_ptr xp;
      mp_size_t rn, newrn;
      mp_size_t sizes[NPOWS], *sizp;
      mp_limb_t di;

      /* Compute the computation precisions from highest to lowest, leaving the
        base case size in 'rn'.  */
      sizp = sizes;
      for (rn = n; ABOVE_THRESHOLD (rn, BINV_NEWTON_THRESHOLD); rn = (rn + 1) >> 1)
        *sizp++ = rn;

      xp = scratch;

      /* Compute a base value of rn limbs.  */
      MPN_ZERO (xp, rn);
      xp[0] = 1;
      binvert_limb (di, up[0]);
      if (BELOW_THRESHOLD (rn, DC_BDIV_Q_THRESHOLD))
        gpmpn_sbpi1_bdiv_q (rp, xp, rn, up, rn, -di);
      else
        gpmpn_dcpi1_bdiv_q (rp, xp, rn, up, rn, -di);

      gpmpn_neg (rp, rp, rn);

      /* Use Newton iterations to get the desired precision.  */
      for (; rn < n; rn = newrn)
      {
        mp_size_t m;
        newrn = *--sizp;

        /* X <- UR. */
        m = gpmpn_mulmod_bnm1_next_size (newrn);
        gpmpn_mulmod_bnm1 (xp, m, up, newrn, rp, rn, xp + m);
        /* Only the values in the range xp + rn .. xp + newrn - 1 are
        used by the _mullo_n below.
        Since m >= newrn, we do not need the following. */
        /* gpmpn_sub_1 (xp + m, xp, rn - (m - newrn), 1); */

        /* R = R(X/B^rn) */
        gpmpn_mullo_n (rp + rn, rp, xp + rn, newrn - rn);
        gpmpn_neg (rp + rn, rp + rn, newrn - rn);
      }
    }

  }
}