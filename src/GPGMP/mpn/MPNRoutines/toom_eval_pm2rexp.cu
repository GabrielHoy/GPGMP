/* gpmpn_toom_eval_pm2rexp -- Evaluate a polynomial in +2^-k and -2^-k

   Contributed to the GNU project by Marco Bodrato

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009 Free Software Foundation, Inc.

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

#if HAVE_NATIVE_gpmpn_addlsh_n
#define DO_gpmpn_addlsh_n(dst, src, n, s, ws) gpmpn_addlsh_n(dst, dst, src, n, s)
#else
    ANYCALLER static mp_limb_t DO_gpmpn_addlsh_n(mp_ptr dst, mp_srcptr src, mp_size_t n, unsigned int s, mp_ptr ws)
    {
#if USE_MUL_1 && 0
      return gpmpn_addmul_1(dst, src, n, CNST_LIMB(1) << (s));
#else
      mp_limb_t __cy;
      __cy = gpmpn_lshift(ws, src, n, s);
      return __cy + gpmpn_add_n(dst, dst, ws, n);
#endif
    }
#endif

    /* Evaluates a polynomial of degree k >= 3. */
    ANYCALLER int gpmpn_toom_eval_pm2rexp(mp_ptr rp, mp_ptr rm, unsigned int q, mp_srcptr ap, mp_size_t n, mp_size_t t, unsigned int s, mp_ptr ws)
    {
      unsigned int i;
      int neg;
      /* {ap,q*n+t} -> {rp,n+1} {rm,n+1} , with {ws, n+1}*/
      ASSERT(n >= t);
      ASSERT(s != 0); /* or _eval_pm1 should be used */
      ASSERT(q > 1);
      ASSERT(s * q < GMP_NUMB_BITS);
      rp[n] = gpmpn_lshift(rp, ap, n, s * q);
      ws[n] = gpmpn_lshift(ws, ap + n, n, s * (q - 1));
      if ((q & 1) != 0)
      {
        ASSERT_NOCARRY(gpmpn_add(ws, ws, n + 1, ap + n * q, t));
        rp[n] += DO_gpmpn_addlsh_n(rp, ap + n * (q - 1), n, s, rm);
      }
      else
      {
        ASSERT_NOCARRY(gpmpn_add(rp, rp, n + 1, ap + n * q, t));
      }
      for (i = 2; i < q - 1; i++)
      {
        rp[n] += DO_gpmpn_addlsh_n(rp, ap + n * i, n, s * (q - i), rm);
        i++;
        ws[n] += DO_gpmpn_addlsh_n(ws, ap + n * i, n, s * (q - i), rm);
      };

      neg = (gpmpn_cmp(rp, ws, n + 1) < 0) ? ~0 : 0;

#if HAVE_NATIVE_gpmpn_add_n_sub_n
      if (neg)
        gpmpn_add_n_sub_n(rp, rm, ws, rp, n + 1);
      else
        gpmpn_add_n_sub_n(rp, rm, rp, ws, n + 1);
#else  /* !HAVE_NATIVE_gpmpn_add_n_sub_n */
      if (neg)
        gpmpn_sub_n(rm, ws, rp, n + 1);
      else
        gpmpn_sub_n(rm, rp, ws, n + 1);

      ASSERT_NOCARRY(gpmpn_add_n(rp, rp, ws, n + 1));
#endif /* !HAVE_NATIVE_gpmpn_add_n_sub_n */

      return neg;
    }

  }
}