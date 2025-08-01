/* gpmpn_dcpi1_bdiv_q -- divide-and-conquer Hensel division with precomputed
   inverse, returning quotient.

   Contributed to the GNU project by Niels Möller and Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2007, 2009-2011, 2017 Free Software Foundation, Inc.

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


    #if 0				/* unused, so leave out for now */
    static mp_size_t
    gpmpn_dcpi1_bdiv_q_n_itch (mp_size_t n)
    {
      /* NOTE: Depends on mullo_n and gpmpn_dcpi1_bdiv_qr_n interface */
      return n;
    }
    #endif

    /* Computes Q = - N / D mod B^n, destroys N.

      N = {np,n}
      D = {dp,n}
    */
    ANYCALLER static void gpmpn_dcpi1_bdiv_q_n (mp_ptr qp, mp_ptr np, mp_srcptr dp, mp_size_t n, mp_limb_t dinv, mp_ptr tp)
    {
      while (ABOVE_THRESHOLD (n, DC_BDIV_Q_THRESHOLD))
      {
        mp_size_t lo, hi;
        mp_limb_t cy;

        lo = n >> 1;			/* floor(n/2) */
        hi = n - lo;			/* ceil(n/2) */

        cy = gpmpn_dcpi1_bdiv_qr_n (qp, np, dp, lo, dinv, tp);

        gpmpn_mullo_n (tp, qp, dp + hi, lo);
        gpmpn_add_n (np + hi, np + hi, tp, lo);

        if (lo < hi)
        {
          cy += gpmpn_addmul_1 (np + lo, qp, lo, dp[lo]);
          np[n - 1] += cy;
        }
        qp += lo;
        np += lo;
        n -= lo;
      }
      gpmpn_sbpi1_bdiv_q (qp, np, n, dp, n, dinv);
    }

    /* Computes Q = - N / D mod B^nn, destroys N.

      N = {np,nn}
      D = {dp,dn}
    */
    ANYCALLER void gpmpn_dcpi1_bdiv_q (mp_ptr qp, mp_ptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn, mp_limb_t dinv)
    {
      mp_size_t qn;
      mp_limb_t cy;
      mp_ptr tp;
      TMP_DECL;

      TMP_MARK;

      ASSERT (dn >= 2);
      ASSERT (nn - dn >= 0);
      ASSERT (dp[0] & 1);

      tp = TMP_SALLOC_LIMBS (dn);

      qn = nn;

      if (qn > dn)
      {
        /* Reduce qn mod dn in a super-efficient manner.  */
        do
          qn -= dn;
        while (qn > dn);

        /* Perform the typically smaller block first.  */
        if (BELOW_THRESHOLD (qn, DC_BDIV_QR_THRESHOLD))
          cy = gpmpn_sbpi1_bdiv_qr (qp, np, 2 * qn, dp, qn, dinv);
        else
          cy = gpmpn_dcpi1_bdiv_qr_n (qp, np, dp, qn, dinv, tp);

        if (qn != dn)
        {
          if (qn > dn - qn)
            gpmpn_mul (tp, qp, qn, dp + qn, dn - qn);
          else
            gpmpn_mul (tp, dp + qn, dn - qn, qp, qn);
          gpmpn_incr_u (tp + qn, cy);

          gpmpn_add (np + qn, np + qn, nn - qn, tp, dn);
          cy = 0;
        }

        np += qn;
        qp += qn;

        qn = nn - qn;
        while (qn > dn)
        {
          gpmpn_add_1 (np + dn, np + dn, qn - dn, cy);
          cy = gpmpn_dcpi1_bdiv_qr_n (qp, np, dp, dn, dinv, tp);
          qp += dn;
          np += dn;
          qn -= dn;
        }
        gpmpn_dcpi1_bdiv_q_n (qp, np, dp, dn, dinv, tp);
      }
      else
      {
        if (BELOW_THRESHOLD (qn, DC_BDIV_Q_THRESHOLD))
          gpmpn_sbpi1_bdiv_q (qp, np, qn, dp, qn, dinv);
        else
          gpmpn_dcpi1_bdiv_q_n (qp, np, dp, qn, dinv, tp);
      }

      TMP_FREE;
    }

  }
}