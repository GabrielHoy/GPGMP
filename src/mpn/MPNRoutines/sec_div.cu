/* mpn_sec_div_qr, mpn_sec_div_r -- Compute Q = floor(U / V), U = U mod V.
   Side-channel silent under the assumption that the used instructions are
   side-channel silent.

   Contributed to the GNU project by Torbjörn Granlund.

Copyright 2011-2015 Free Software Foundation, Inc.

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

/*
  GPGMP NOTES - sec_div.cuh
  "Multi Function Files"

  (see the 'configuration' file in GMP source for some useful explanations, the section on 'adding new mpn function files' was useful to read here...)

  the original version of this file is referred to by GMP as a 'multi-function file'

  functions in these files are odd - they have #if macros at the top that define the function name and return type along with some other...
  ...things, then a "generic" function is written in the file whose behavior changes depending on the path of those previous #if statements and thereby which "function name" it is running under...
  ...- to make these files work well without using GMP's makefile and configuration system, we need to undo this dynamic behavior, we can do...
  ...that by simply creating duplicates of the function definition and replacing any related details or #if macro logic branches accordingly...
  ...Doing this is of course less 'elegant' than GMP's approach, but it accomplishes the same goal and shouldn't have any negative implications for end users (and let's face it my code is generally probably not as elegant as GMP's to begin with haha.)

*/



#include "gpgmp.cuh"
#include "longlong.cuh"
#include "gmp.h"

namespace gpgmp
{
  namespace mpnRoutines
  {

    ANYCALLER mp_size_t mpn_sec_div_qr_itch(mp_size_t nn, mp_size_t dn)
    {
      /* Needs (nn + dn + 1) + mpn_sec_pi1_div_qr's needs of (2nn' - dn + 1) for a
         total of 3nn + 4 limbs at tp.  Note that mpn_sec_pi1_div_qr's nn is one
         greater than ours, therefore +4 and not just +2.  */
      return 3 * nn + 4;
    }

    mp_limb_t mpn_sec_div_qr(mp_ptr qp, mp_ptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn, mp_ptr tp)
    {
      mp_limb_t d1, d0;
      unsigned int cnt;
      mp_limb_t inv32;

      ASSERT(dn >= 1);
      ASSERT(nn >= dn);
      ASSERT(dp[dn - 1] != 0);

      d1 = dp[dn - 1];
      count_leading_zeros(cnt, d1);

      if (cnt != 0)
      {
        mp_limb_t qh, cy;
        mp_ptr np2, dp2;
        dp2 = tp; /* dn limbs */
        mpn_lshift(dp2, dp, dn, cnt);

        np2 = tp + dn; /* (nn + 1) limbs */
        cy = mpn_lshift(np2, np, nn, cnt);
        np2[nn++] = cy;

        d0 = dp2[dn - 1];
        d0 += (~d0 != 0);
        invert_limb(inv32, d0);

        /* We add nn + dn to tp here, not nn + 1 + dn, as expected.  This is
     since nn here will have been incremented.  */
        qh = mpn_sec_pi1_div_qr(np2 + dn, np2, nn, dp2, dn, inv32, tp + nn + dn);
        ASSERT(qh == 0); /* FIXME: this indicates inefficiency! */
        MPN_COPY(qp, np2 + dn, nn - dn - 1);
        qh = np2[nn - 1];

        mpn_rshift(np, np2, dn, cnt);

        return qh;
      }
      else
      {
        /* FIXME: Consider copying np => np2 here, adding a 0-limb at the top.
     That would simplify the underlying pi1 function, since then it could
     assume nn > dn.  */
        d0 = dp[dn - 1];
        d0 += (~d0 != 0);
        invert_limb(inv32, d0);

        return mpn_sec_pi1_div_qr(qp, np, nn, dp, dn, inv32, tp);
      }
    }






















    ANYCALLER mp_size_t mpn_sec_div_r_itch(mp_size_t nn, mp_size_t dn)
    {
      /* Needs (nn + dn + 1) + mpn_sec_pi1_div_r's needs of (dn + 1) for a total of
         nn + 2dn + 2 limbs at tp.  */
      return nn + 2 * dn + 2;
    }

    void
    mpn_sec_div_r(
              mp_ptr np,
          mp_size_t nn,
          mp_srcptr dp, mp_size_t dn,
          mp_ptr tp)
    {
      mp_limb_t d1, d0;
      unsigned int cnt;
      mp_limb_t inv32;

      ASSERT(dn >= 1);
      ASSERT(nn >= dn);
      ASSERT(dp[dn - 1] != 0);

      d1 = dp[dn - 1];
      count_leading_zeros(cnt, d1);

      if (cnt != 0)
      {
        mp_limb_t qh, cy;
        mp_ptr np2, dp2;
        dp2 = tp; /* dn limbs */
        mpn_lshift(dp2, dp, dn, cnt);

        np2 = tp + dn; /* (nn + 1) limbs */
        cy = mpn_lshift(np2, np, nn, cnt);
        np2[nn++] = cy;

        d0 = dp2[dn - 1];
        d0 += (~d0 != 0);
        invert_limb(inv32, d0);

        /* We add nn + dn to tp here, not nn + 1 + dn, as expected.  This is
     since nn here will have been incremented.  */
        mpn_sec_pi1_div_r(np2, nn, dp2, dn, inv32, tp + nn + dn);

        mpn_rshift(np, np2, dn, cnt);
      }
      else
      {
        /* FIXME: Consider copying np => np2 here, adding a 0-limb at the top.
     That would simplify the underlying pi1 function, since then it could
     assume nn > dn.  */
        d0 = dp[dn - 1];
        d0 += (~d0 != 0);
        invert_limb(inv32, d0);

        mpn_sec_pi1_div_r(np, nn, dp, dn, inv32, tp);
      }
    }

  }
}