/* gpmpn_sqr_basecase -- Internal routine to square a natural number
   of length n.

   THIS IS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH THIS FUNCTION THROUGH DOCUMENTED INTERFACES.


Copyright 1991-1994, 1996, 1997, 2000-2005, 2008, 2010, 2011, 2017 Free
Software Foundation, Inc.

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

#if HAVE_NATIVE_gpmpn_sqr_diagonal
#define MPN_SQR_DIAGONAL(rp, up, n) \
  gpmpn_sqr_diagonal(rp, up, n)
#else
#define MPN_SQR_DIAGONAL(rp, up, n)                              \
  do                                                             \
  {                                                              \
    mp_size_t _i;                                                \
    for (_i = 0; _i < (n); _i++)                                 \
    {                                                            \
      mp_limb_t ul, lpl;                                         \
      ul = (up)[_i];                                             \
      umul_ppmm((rp)[2 * _i + 1], lpl, ul, ul << GMP_NAIL_BITS); \
      (rp)[2 * _i] = lpl >> GMP_NAIL_BITS;                       \
    }                                                            \
  } while (0)
#endif

#if HAVE_NATIVE_gpmpn_sqr_diag_addlsh1
#define MPN_SQR_DIAG_ADDLSH1(rp, tp, up, n) \
  gpmpn_sqr_diag_addlsh1(rp, tp, up, n)
#else
#if HAVE_NATIVE_gpmpn_addlsh1_n
#define MPN_SQR_DIAG_ADDLSH1(rp, tp, up, n)            \
  do                                                   \
  {                                                    \
    mp_limb_t cy;                                      \
    MPN_SQR_DIAGONAL(rp, up, n);                       \
    cy = gpmpn_addlsh1_n(rp + 1, rp + 1, tp, 2 * n - 2); \
    rp[2 * n - 1] += cy;                               \
  } while (0)
#else
#define MPN_SQR_DIAG_ADDLSH1(rp, tp, up, n)         \
  do                                                \
  {                                                 \
    mp_limb_t cy;                                   \
    MPN_SQR_DIAGONAL(rp, up, n);                    \
    cy = gpmpn_lshift(tp, tp, 2 * n - 2, 1);          \
    cy += gpmpn_add_n(rp + 1, rp + 1, tp, 2 * n - 2); \
    rp[2 * n - 1] += cy;                            \
  } while (0)
#endif
#endif

#undef READY_WITH_gpmpn_sqr_basecase

#if !defined(READY_WITH_gpmpn_sqr_basecase) && HAVE_NATIVE_gpmpn_addmul_2s
    ANYCALLER void
    gpmpn_sqr_basecase(mp_ptr rp, mp_srcptr up, mp_size_t n)
    {
      mp_size_t i;
      mp_limb_t tarr[2 * SQR_TOOM2_THRESHOLD];
      mp_ptr tp = tarr;
      mp_limb_t cy;

      /* must fit 2*n limbs in tarr */
      ASSERT(n <= SQR_TOOM2_THRESHOLD);

      if ((n & 1) != 0)
      {
        if (n == 1)
        {
          mp_limb_t ul, lpl;
          ul = up[0];
          umul_ppmm(rp[1], lpl, ul, ul << GMP_NAIL_BITS);
          rp[0] = lpl >> GMP_NAIL_BITS;
          return;
        }

        MPN_ZERO(tp, n);

        for (i = 0; i <= n - 2; i += 2)
        {
          cy = gpmpn_addmul_2s(tp + 2 * i, up + i + 1, n - (i + 1), up + i);
          tp[n + i] = cy;
        }
      }
      else
      {
        if (n == 2)
        {
#if HAVE_NATIVE_gpmpn_mul_2
          rp[3] = gpmpn_mul_2(rp, up, 2, up);
#else
          rp[0] = 0;
          rp[1] = 0;
          rp[3] = gpmpn_addmul_2(rp, up, 2, up);
#endif
          return;
        }

        MPN_ZERO(tp, n);

        for (i = 0; i <= n - 4; i += 2)
        {
          cy = gpmpn_addmul_2s(tp + 2 * i, up + i + 1, n - (i + 1), up + i);
          tp[n + i] = cy;
        }
        cy = gpmpn_addmul_1(tp + 2 * n - 4, up + n - 1, 1, up[n - 2]);
        tp[2 * n - 3] = cy;
      }

      MPN_SQR_DIAG_ADDLSH1(rp, tp, up, n);
    }
#define READY_WITH_gpmpn_sqr_basecase
#endif

#if !defined(READY_WITH_gpmpn_sqr_basecase) && HAVE_NATIVE_gpmpn_addmul_2

    /* gpmpn_sqr_basecase using plain gpmpn_addmul_2.

       This is tricky, since we have to let gpmpn_addmul_2 make some undesirable
       multiplies, u[k]*u[k], that we would like to let gpmpn_sqr_diagonal handle.
       This forces us to conditionally add or subtract the gpmpn_sqr_diagonal
       results.  Examples of the product we form:

       n = 4              n = 5		n = 6
       u1u0 * u3u2u1      u1u0 * u4u3u2u1	u1u0 * u5u4u3u2u1
       u2 * u3	      u3u2 * u4u3	u3u2 * u5u4u3
              u4 * u5
       add: u0 u2 u3      add: u0 u2 u4	add: u0 u2 u4 u5
       sub: u1	      sub: u1 u3	sub: u1 u3
    */

    ANYCALLER void
    gpmpn_sqr_basecase(mp_ptr rp, mp_srcptr up, mp_size_t n)
    {
      mp_size_t i;
      mp_limb_t tarr[2 * SQR_TOOM2_THRESHOLD];
      mp_ptr tp = tarr;
      mp_limb_t cy;

      /* must fit 2*n limbs in tarr */
      ASSERT(n <= SQR_TOOM2_THRESHOLD);

      if ((n & 1) != 0)
      {
        mp_limb_t x0, x1;

        if (n == 1)
        {
          mp_limb_t ul, lpl;
          ul = up[0];
          umul_ppmm(rp[1], lpl, ul, ul << GMP_NAIL_BITS);
          rp[0] = lpl >> GMP_NAIL_BITS;
          return;
        }

        /* The code below doesn't like unnormalized operands.  Since such
     operands are unusual, handle them with a dumb recursion.  */
        if (up[n - 1] == 0)
        {
          rp[2 * n - 2] = 0;
          rp[2 * n - 1] = 0;
          gpmpn_sqr_basecase(rp, up, n - 1);
          return;
        }

        MPN_ZERO(tp, n);

        for (i = 0; i <= n - 2; i += 2)
        {
          cy = gpmpn_addmul_2(tp + 2 * i, up + i + 1, n - (i + 1), up + i);
          tp[n + i] = cy;
        }

        MPN_SQR_DIAGONAL(rp, up, n);

        for (i = 2;; i += 4)
        {
          x0 = rp[i + 0];
          rp[i + 0] = (-x0) & GMP_NUMB_MASK;
          x1 = rp[i + 1];
          rp[i + 1] = (-x1 - (x0 != 0)) & GMP_NUMB_MASK;
          __GMPN_SUB_1(cy, rp + i + 2, rp + i + 2, 2, (x1 | x0) != 0);
          if (i + 4 >= 2 * n)
            break;
          gpmpn_incr_u(rp + i + 4, cy);
        }
      }
      else
      {
        mp_limb_t x0, x1;

        if (n == 2)
        {
#if HAVE_NATIVE_gpmpn_mul_2
          rp[3] = gpmpn_mul_2(rp, up, 2, up);
#else
          rp[0] = 0;
          rp[1] = 0;
          rp[3] = gpmpn_addmul_2(rp, up, 2, up);
#endif
          return;
        }

        /* The code below doesn't like unnormalized operands.  Since such
     operands are unusual, handle them with a dumb recursion.  */
        if (up[n - 1] == 0)
        {
          rp[2 * n - 2] = 0;
          rp[2 * n - 1] = 0;
          gpmpn_sqr_basecase(rp, up, n - 1);
          return;
        }

        MPN_ZERO(tp, n);

        for (i = 0; i <= n - 4; i += 2)
        {
          cy = gpmpn_addmul_2(tp + 2 * i, up + i + 1, n - (i + 1), up + i);
          tp[n + i] = cy;
        }
        cy = gpmpn_addmul_1(tp + 2 * n - 4, up + n - 1, 1, up[n - 2]);
        tp[2 * n - 3] = cy;

        MPN_SQR_DIAGONAL(rp, up, n);

        for (i = 2;; i += 4)
        {
          x0 = rp[i + 0];
          rp[i + 0] = (-x0) & GMP_NUMB_MASK;
          x1 = rp[i + 1];
          rp[i + 1] = (-x1 - (x0 != 0)) & GMP_NUMB_MASK;
          if (i + 6 >= 2 * n)
            break;
          __GMPN_SUB_1(cy, rp + i + 2, rp + i + 2, 2, (x1 | x0) != 0);
          gpmpn_incr_u(rp + i + 4, cy);
        }
        gpmpn_decr_u(rp + i + 2, (x1 | x0) != 0);
      }

#if HAVE_NATIVE_gpmpn_addlsh1_n
      cy = gpmpn_addlsh1_n(rp + 1, rp + 1, tp, 2 * n - 2);
#else
      cy = gpmpn_lshift(tp, tp, 2 * n - 2, 1);
      cy += gpmpn_add_n(rp + 1, rp + 1, tp, 2 * n - 2);
#endif
      rp[2 * n - 1] += cy;
    }
#define READY_WITH_gpmpn_sqr_basecase
#endif

#if !defined(READY_WITH_gpmpn_sqr_basecase) && HAVE_NATIVE_gpmpn_sqr_diag_addlsh1

    /* gpmpn_sqr_basecase using gpmpn_addmul_1 and gpmpn_sqr_diag_addlsh1, avoiding stack
       allocation.  */
    ANYCALLER void
    gpmpn_sqr_basecase(mp_ptr rp, mp_srcptr up, mp_size_t n)
    {
      if (n == 1)
      {
        mp_limb_t ul, lpl;
        ul = up[0];
        umul_ppmm(rp[1], lpl, ul, ul << GMP_NAIL_BITS);
        rp[0] = lpl >> GMP_NAIL_BITS;
      }
      else
      {
        mp_size_t i;
        mp_ptr xp;

        rp += 1;
        rp[n - 1] = gpmpn_mul_1(rp, up + 1, n - 1, up[0]);
        for (i = n - 2; i != 0; i--)
        {
          up += 1;
          rp += 2;
          rp[i] = gpmpn_addmul_1(rp, up + 1, i, up[0]);
        }

        xp = rp - 2 * n + 3;
        gpmpn_sqr_diag_addlsh1(xp, xp + 1, up - n + 2, n);
      }
    }
#define READY_WITH_gpmpn_sqr_basecase
#endif

#if !defined(READY_WITH_gpmpn_sqr_basecase)

    /* Default gpmpn_sqr_basecase using gpmpn_addmul_1.  */
    ANYCALLER void gpmpn_sqr_basecase(mp_ptr rp, mp_srcptr up, mp_size_t n)
    {
      mp_size_t i;

      ASSERT(n >= 1);
      ASSERT(!MPN_OVERLAP_P(rp, 2 * n, up, n));

      if (n == 1)
      {
        mp_limb_t ul, lpl;
        ul = up[0];
        umul_ppmm(rp[1], lpl, ul, ul << GMP_NAIL_BITS);
        rp[0] = lpl >> GMP_NAIL_BITS;
      }
      else
      {
        mp_limb_t tarr[2 * SQR_TOOM2_THRESHOLD];
        mp_ptr tp = tarr;
        mp_limb_t cy;

        /* must fit 2*n limbs in tarr */
        ASSERT(n <= SQR_TOOM2_THRESHOLD);

        cy = gpmpn_mul_1(tp, up + 1, n - 1, up[0]);
        tp[n - 1] = cy;
        for (i = 2; i < n; i++)
        {
          mp_limb_t cy;
          cy = gpmpn_addmul_1(tp + 2 * i - 2, up + i, n - i, up[i - 1]);
          tp[n + i - 2] = cy;
        }

        MPN_SQR_DIAG_ADDLSH1(rp, tp, up, n);
      }
    }

    ANYCALLER void gpmpn_sqr_basecase_with_preallocated_tarr(mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t* tarr)
    {
      mp_size_t i;

      ASSERT(n >= 1);
      ASSERT(!MPN_OVERLAP_P(rp, 2 * n, up, n));

      if (n == 1)
      {
        mp_limb_t ul, lpl;
        ul = up[0];
        umul_ppmm(rp[1], lpl, ul, ul << GMP_NAIL_BITS);
        rp[0] = lpl >> GMP_NAIL_BITS;
      }
      else
      {
        //mp_limb_t tarr[2 * SQR_TOOM2_THRESHOLD];
        mp_ptr tp = tarr;
        mp_limb_t cy;

        /* must fit 2*n limbs in tarr */
        ASSERT(n <= SQR_TOOM2_THRESHOLD);

        cy = gpmpn_mul_1(tp, up + 1, n - 1, up[0]);
        tp[n - 1] = cy;
        for (i = 2; i < n; i++)
        {
          mp_limb_t cy;
          cy = gpmpn_addmul_1(tp + 2 * i - 2, up + i, n - i, up[i - 1]);
          tp[n + i - 2] = cy;
        }

        MPN_SQR_DIAG_ADDLSH1(rp, tp, up, n);
      }
    }
#define READY_WITH_gpmpn_sqr_basecase
#endif

  }
}