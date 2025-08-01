/* gpmpn_sec_invert

   Contributed to the GNU project by Niels Möller

Copyright 2013 Free Software Foundation, Inc.

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

    /* FIXME: Ought to return carry */
    ANYCALLER static void gpmpn_cnd_neg(int cnd, mp_limb_t *rp, const mp_limb_t *ap, mp_size_t n, mp_ptr scratch)
    {
      gpmpn_lshift(scratch, ap, n, 1);
      gpmpn_cnd_sub_n(cnd, rp, ap, scratch, n);
    }

    ANYCALLER static int gpmpn_sec_eq_ui(mp_srcptr ap, mp_size_t n, mp_limb_t b)
    {
      mp_limb_t d;
      ASSERT(n > 0);

      d = ap[0] ^ b;

      while (--n > 0)
        d |= ap[n];

      return d == 0;
    }

    ANYCALLER mp_size_t gpmpn_sec_invert_itch(mp_size_t n)
    {
      return 4 * n;
    }

    /* Compute V <-- A^{-1} (mod M), in data-independent time. M must be
       odd. Returns 1 on success, and 0 on failure (i.e., if gcd (A, m) !=
       1). Inputs and outputs of size n, and no overlap allowed. The {ap,
       n} area is destroyed. For arbitrary inputs, bit_size should be
       2*n*GMP_NUMB_BITS, but if A or M are known to be smaller, e.g., if
       M = 2^521 - 1 and A < M, bit_size can be any bound on the sum of
       the bit sizes of A and M. */
    ANYCALLER int gpmpn_sec_invert(mp_ptr vp, mp_ptr ap, mp_srcptr mp, mp_size_t n, mp_bitcnt_t bit_size, mp_ptr scratch)
    {
      ASSERT(n > 0);
      ASSERT(bit_size > 0);
      ASSERT(mp[0] & 1);
      ASSERT(!MPN_OVERLAP_P(ap, n, vp, n));
#define bp (scratch + n)
#define up (scratch + 2 * n)
#define m1hp (scratch + 3 * n)

      /* Maintain

           a = u * orig_a (mod m)
           b = v * orig_a (mod m)

         and b odd at all times. Initially,

           a = a_orig, u = 1
           b = m,      v = 0
         */

      up[0] = 1;
      gpmpn_zero(up + 1, n - 1);
      gpmpn_copyi(bp, mp, n);
      gpmpn_zero(vp, n);

      ASSERT_CARRY(gpmpn_rshift(m1hp, mp, n, 1));
      ASSERT_NOCARRY(gpmpn_sec_add_1(m1hp, m1hp, n, 1, scratch));

      while (bit_size-- > 0)
      {
        mp_limb_t odd, swap, cy;

        /* Always maintain b odd. The logic of the iteration is as
     follows. For a, b:

       odd = a & 1
       a -= odd * b
       if (underflow from a-b)
         {
           b += a, assigns old a
           a = B^n-a
         }

       a /= 2

     For u, v:

       if (underflow from a - b)
         swap u, v
       u -= odd * v
       if (underflow from u - v)
         u += m

       u /= 2
       if (a one bit was shifted out)
         u += (m+1)/2

     As long as a > 0, the quantity

       (bitsize of a) + (bitsize of b)

     is reduced by at least one bit per iteration, hence after (bit_size of
     orig_a) + (bit_size of m) - 1 iterations we surely have a = 0. Then b
     = gcd(orig_a, m) and if b = 1 then also v = orig_a^{-1} (mod m).
        */

        ASSERT(bp[0] & 1);
        odd = ap[0] & 1;

        swap = gpmpn_cnd_sub_n(odd, ap, ap, bp, n);
        gpmpn_cnd_add_n(swap, bp, bp, ap, n);
        gpmpn_cnd_neg(swap, ap, ap, n, scratch);

        gpmpn_cnd_swap(swap, up, vp, n);
        cy = gpmpn_cnd_sub_n(odd, up, up, vp, n);
        cy -= gpmpn_cnd_add_n(cy, up, up, mp, n);
        ASSERT(cy == 0);

        cy = gpmpn_rshift(ap, ap, n, 1);
        ASSERT(cy == 0);
        cy = gpmpn_rshift(up, up, n, 1);
        cy = gpmpn_cnd_add_n(cy, up, up, m1hp, n);
        ASSERT(cy == 0);
      }
      /* Should be all zeros, but check only extreme limbs */
      ASSERT((ap[0] | ap[n - 1]) == 0);
      /* Check if indeed gcd == 1. */
      return gpmpn_sec_eq_ui(bp, n, 1);
#undef bp
#undef up
#undef m1hp
    }

  }
}