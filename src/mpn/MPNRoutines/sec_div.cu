/* gpmpn_sec_div_qr, gpmpn_sec_div_r -- Compute Q = floor(U / V), U = U mod V.
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



#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpnRoutines
  {

    ANYCALLER mp_size_t gpmpn_sec_div_qr_itch(mp_size_t nn, mp_size_t dn)
    {
      /* Needs (nn + dn + 1) + gpmpn_sec_pi1_div_qr's needs of (2nn' - dn + 1) for a
         total of 3nn + 4 limbs at tp.  Note that gpmpn_sec_pi1_div_qr's nn is one
         greater than ours, therefore +4 and not just +2.  */
      return 3 * nn + 4;
    }

    ANYCALLER mp_limb_t gpmpn_sec_div_qr(mp_ptr quotient_ptr, mp_ptr numerator_ptr, mp_size_t numerator_size, mp_srcptr divisor_ptr, mp_size_t divisor_size, mp_ptr temp_ptr)
    {
      mp_limb_t divisor_high, divisor_normalized;
      unsigned int leading_zeros;
      mp_limb_t inverse_32;

      ASSERT(divisor_size >= 1);
      ASSERT(numerator_size >= divisor_size);
      ASSERT(divisor_ptr[divisor_size - 1] != 0);

      divisor_high = divisor_ptr[divisor_size - 1];
      count_leading_zeros(leading_zeros, divisor_high);

      if (leading_zeros != 0)
      {
        mp_limb_t quotient_high, carry;
        mp_ptr normalized_numerator, normalized_divisor;
        normalized_divisor = temp_ptr; /* divisor_size limbs */
        gpmpn_lshift(normalized_divisor, divisor_ptr, divisor_size, leading_zeros);

        normalized_numerator = temp_ptr + divisor_size; /* (numerator_size + 1) limbs */
        carry = gpmpn_lshift(normalized_numerator, numerator_ptr, numerator_size, leading_zeros);
        normalized_numerator[numerator_size++] = carry;

        divisor_normalized = normalized_divisor[divisor_size - 1];
        divisor_normalized += (~divisor_normalized != 0);
        invert_limb(inverse_32, divisor_normalized);

        /* We add numerator_size + divisor_size to temp_ptr here, not numerator_size + 1 + divisor_size, as expected.  This is
           since numerator_size here will have been incremented.  */
        quotient_high = gpmpn_sec_pi1_div_qr(normalized_numerator + divisor_size, normalized_numerator, numerator_size, normalized_divisor, divisor_size, inverse_32, temp_ptr + numerator_size + divisor_size);
        ASSERT(quotient_high == 0); /* FIXME: this indicates inefficiency! */
        MPN_COPY(quotient_ptr, normalized_numerator + divisor_size, numerator_size - divisor_size - 1);
        quotient_high = normalized_numerator[numerator_size - 1];

        gpmpn_rshift(numerator_ptr, normalized_numerator, divisor_size, leading_zeros);

        return quotient_high;
      }
      else
      {
        /* FIXME: Consider copying numerator_ptr => normalized_numerator here, adding a 0-limb at the top.
           That would simplify the underlying pi1 function, since then it could
           assume numerator_size > divisor_size.  */
        divisor_normalized = divisor_ptr[divisor_size - 1];
        divisor_normalized += (~divisor_normalized != 0);
        invert_limb(inverse_32, divisor_normalized);

        return gpmpn_sec_pi1_div_qr(quotient_ptr, numerator_ptr, numerator_size, divisor_ptr, divisor_size, inverse_32, temp_ptr);
      }
    }






















    ANYCALLER mp_size_t gpmpn_sec_div_r_itch(mp_size_t nn, mp_size_t dn)
    {
      /* Needs (nn + dn + 1) + gpmpn_sec_pi1_div_r's needs of (dn + 1) for a total of
         nn + 2dn + 2 limbs at tp.  */
      return nn + 2 * dn + 2;
    }

    void
    gpmpn_sec_div_r(
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
        gpmpn_lshift(dp2, dp, dn, cnt);

        np2 = tp + dn; /* (nn + 1) limbs */
        cy = gpmpn_lshift(np2, np, nn, cnt);
        np2[nn++] = cy;

        d0 = dp2[dn - 1];
        d0 += (~d0 != 0);
        invert_limb(inv32, d0);

        /* We add nn + dn to tp here, not nn + 1 + dn, as expected.  This is
     since nn here will have been incremented.  */
        gpmpn_sec_pi1_div_r(np2, nn, dp2, dn, inv32, tp + nn + dn);

        gpmpn_rshift(np, np2, dn, cnt);
      }
      else
      {
        /* FIXME: Consider copying np => np2 here, adding a 0-limb at the top.
     That would simplify the underlying pi1 function, since then it could
     assume nn > dn.  */
        d0 = dp[dn - 1];
        d0 += (~d0 != 0);
        invert_limb(inv32, d0);

        gpmpn_sec_pi1_div_r(np, nn, dp, dn, inv32, tp);
      }
    }

  }
}