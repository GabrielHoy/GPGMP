/* gpmpn_toom3_sqr -- Square {ap,an}.

   Contributed to the GNU project by Torbjorn Granlund.
   Additional improvements by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006-2010, 2012, 2015, 2021 Free Software Foundation, Inc.

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

    /* Evaluate in: -1, 0, +1, +2, +inf

      <-s--><--n--><--n-->
       ____ ______ ______
      |_a2_|___a1_|___a0_|

      v0  =  a0         ^2 #   A(0)^2
      v1  = (a0+ a1+ a2)^2 #   A(1)^2    ah  <= 2
      vm1 = (a0- a1+ a2)^2 #  A(-1)^2   |ah| <= 1
      v2  = (a0+2a1+4a2)^2 #   A(2)^2    ah  <= 6
      vinf=          a2 ^2 # A(inf)^2
    */

#if TUNE_PROGRAM_BUILD || WANT_FAT_BINARY
#define MAYBE_sqr_basecase 1
#define MAYBE_sqr_toom3 1
#else
#define MAYBE_sqr_basecase \
  (SQR_TOOM3_THRESHOLD < 3 * SQR_TOOM2_THRESHOLD)
#define MAYBE_sqr_toom3 \
  (SQR_TOOM4_THRESHOLD >= 3 * SQR_TOOM3_THRESHOLD)
#endif

#define TOOM3_SQR_REC(p, a, n, ws)                                        \
  do                                                                      \
  {                                                                       \
    if (MAYBE_sqr_basecase && BELOW_THRESHOLD(n, SQR_TOOM2_THRESHOLD))    \
      gpmpn_sqr_basecase(p, a, n);                                          \
    else if (!MAYBE_sqr_toom3 || BELOW_THRESHOLD(n, SQR_TOOM3_THRESHOLD)) \
      gpmpn_toom2_sqr(p, a, n, ws);                                         \
    else                                                                  \
      gpmpn_toom3_sqr(p, a, n, ws);                                         \
  } while (0)

    ANYCALLER void
    gpmpn_toom3_sqr(mp_ptr pp,
                  mp_srcptr ap, mp_size_t an,
                  mp_ptr scratch)
    {
      const int __gpmpn_cpuvec_initialized = 1;
      mp_size_t n, s;
      mp_limb_t cy, vinf0;
      mp_ptr gp;
      mp_ptr as1, asm1, as2;

#define a0 ap
#define a1 (ap + n)
#define a2 (ap + 2 * n)

      n = (an + 2) / (size_t)3;

      s = an - 2 * n;

      ASSERT(0 < s && s <= n);

      as1 = scratch + 4 * n + 4;
      asm1 = scratch + 2 * n + 2;
      as2 = pp + n + 1;

      gp = scratch;

      /* Compute as1 and asm1.  */
      cy = gpmpn_add(gp, a0, n, a2, s);
#if HAVE_NATIVE_gpmpn_add_n_sub_n
      if (cy == 0 && gpmpn_cmp(gp, a1, n) < 0)
      {
        cy = gpmpn_add_n_sub_n(as1, asm1, a1, gp, n);
        as1[n] = cy >> 1;
        asm1[n] = 0;
      }
      else
      {
        mp_limb_t cy2;
        cy2 = gpmpn_add_n_sub_n(as1, asm1, gp, a1, n);
        as1[n] = cy + (cy2 >> 1);
        asm1[n] = cy - (cy2 & 1);
      }
#else
      as1[n] = cy + gpmpn_add_n(as1, gp, a1, n);
      if (cy == 0 && gpmpn_cmp(gp, a1, n) < 0)
      {
        gpmpn_sub_n(asm1, a1, gp, n);
        asm1[n] = 0;
      }
      else
      {
        cy -= gpmpn_sub_n(asm1, gp, a1, n);
        asm1[n] = cy;
      }
#endif

      /* Compute as2.  */
#if HAVE_NATIVE_gpmpn_rsblsh1_n
      cy = gpmpn_add_n(as2, a2, as1, s);
      if (s != n)
        cy = gpmpn_add_1(as2 + s, as1 + s, n - s, cy);
      cy += as1[n];
      cy = 2 * cy + gpmpn_rsblsh1_n(as2, a0, as2, n);
#else
#if HAVE_NATIVE_gpmpn_addlsh1_n
      cy = gpmpn_addlsh1_n(as2, a1, a2, s);
      if (s != n)
        cy = gpmpn_add_1(as2 + s, a1 + s, n - s, cy);
      cy = 2 * cy + gpmpn_addlsh1_n(as2, a0, as2, n);
#else
      cy = gpmpn_add_n(as2, a2, as1, s);
      if (s != n)
        cy = gpmpn_add_1(as2 + s, as1 + s, n - s, cy);
      cy += as1[n];
      cy = 2 * cy + gpmpn_lshift(as2, as2, n, 1);
      cy -= gpmpn_sub_n(as2, as2, a0, n);
#endif
#endif
      as2[n] = cy;

      ASSERT(as1[n] <= 2);
      ASSERT(asm1[n] <= 1);

#define v0 pp                    /* 2n */
#define v1 (pp + 2 * n)          /* 2n+1 */
#define vinf (pp + 4 * n)        /* s+s */
#define vm1 scratch              /* 2n+1 */
#define v2 (scratch + 2 * n + 1) /* 2n+2 */
#define scratch_out (scratch + 5 * n + 5)

      /* vm1, 2n+1 limbs */
#ifdef SMALLER_RECURSION
      TOOM3_SQR_REC(vm1, asm1, n, scratch_out);
      cy = asm1[n];
      if (cy != 0)
      {
#if HAVE_NATIVE_gpmpn_addlsh1_n_ip1
        cy += gpmpn_addlsh1_n_ip1(vm1 + n, asm1, n);
#else
        cy += gpmpn_addmul_1(vm1 + n, asm1, n, CNST_LIMB(2));
#endif
      }
      vm1[2 * n] = cy;
#else
      vm1[2 * n] = 0;
      TOOM3_SQR_REC(vm1, asm1, n + asm1[n], scratch_out);
#endif

      TOOM3_SQR_REC(v2, as2, n + 1, scratch_out); /* v2, 2n+1 limbs */

      TOOM3_SQR_REC(vinf, a2, s, scratch_out); /* vinf, s+s limbs */

      vinf0 = vinf[0]; /* v1 overlaps with this */

#ifdef SMALLER_RECURSION
      /* v1, 2n+1 limbs */
      TOOM3_SQR_REC(v1, as1, n, scratch_out);
      cy = as1[n];
      if (cy == 1)
      {
#if HAVE_NATIVE_gpmpn_addlsh1_n_ip1
        cy += gpmpn_addlsh1_n_ip1(v1 + n, as1, n);
#else
        cy += gpmpn_addmul_1(v1 + n, as1, n, CNST_LIMB(2));
#endif
      }
      else if (cy != 0)
      {
#if HAVE_NATIVE_gpmpn_addlsh2_n_ip1
        cy = 4 + gpmpn_addlsh2_n_ip1(v1 + n, as1, n);
#else
        cy = 4 + gpmpn_addmul_1(v1 + n, as1, n, CNST_LIMB(4));
#endif
      }
      v1[2 * n] = cy;
#else
      cy = vinf[1];
      TOOM3_SQR_REC(v1, as1, n + 1, scratch_out);
      vinf[1] = cy;
#endif

      TOOM3_SQR_REC(v0, ap, n, scratch_out); /* v0, 2n limbs */

      gpmpn_toom_interpolate_5pts(pp, v2, vm1, n, s + s, 0, vinf0);
    }

  }
}