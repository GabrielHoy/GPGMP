/* gpmpn_dcpi1_bdiv_qr -- divide-and-conquer Hensel division with precomputed
   inverse, returning quotient and remainder.

   Contributed to the GNU project by Niels Möller and Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2007, 2009, 2010, 2017 Free Software Foundation, Inc.

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


    /* Computes Hensel binary division of {np, 2*n} by {dp, n}.

   Output:

      q = -n * d^{-1} mod 2^{qn * GMP_NUMB_BITS},

      r = (n + q * d) * 2^{-qn * GMP_NUMB_BITS}

   Stores q at qp. Stores the n least significant limbs of r at the high half
   of np, and returns the carry from the addition n + q*d.

   d must be odd. dinv is (-d)^-1 mod 2^GMP_NUMB_BITS. */
    ANYCALLER mp_size_t gpmpn_dcpi1_bdiv_qr_n_itch (mp_size_t n)
    {
      return n; //lolwot
    }

    ANYCALLER mp_limb_t gpmpn_dcpi1_bdiv_qr_n (mp_ptr qp, mp_ptr np, mp_srcptr dp, mp_size_t n, mp_limb_t dinv, mp_ptr tp)
    {
      mp_size_t lo, hi;
      mp_limb_t cy;
      mp_limb_t rh;

      lo = n >> 1;			/* floor(n/2) */
      hi = n - lo;			/* ceil(n/2) */

      if (BELOW_THRESHOLD (lo, DC_BDIV_QR_THRESHOLD))
        cy = gpmpn_sbpi1_bdiv_qr (qp, np, 2 * lo, dp, lo, dinv);
      else
        cy = gpmpn_dcpi1_bdiv_qr_n (qp, np, dp, lo, dinv, tp);

      gpmpn_mul (tp, dp + lo, hi, qp, lo);

      gpmpn_incr_u (tp + lo, cy);
      rh = gpmpn_add (np + lo, np + lo, n + hi, tp, n);

      if (BELOW_THRESHOLD (hi, DC_BDIV_QR_THRESHOLD))
        cy = gpmpn_sbpi1_bdiv_qr (qp + lo, np + lo, 2 * hi, dp, hi, dinv);
      else
        cy = gpmpn_dcpi1_bdiv_qr_n (qp + lo, np + lo, dp, hi, dinv, tp);

      gpmpn_mul (tp, qp + lo, hi, dp + hi, lo);

      gpmpn_incr_u (tp + hi, cy);
      rh += gpmpn_add_n (np + n, np + n, tp, n);

      return rh;
    }

    ANYCALLER mp_limb_t gpmpn_dcpi1_bdiv_qr (mp_ptr qp, mp_ptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn, mp_limb_t dinv)
    {
      mp_size_t qn;
      mp_limb_t rr, cy;
      mp_ptr tp;
      TMP_DECL;

      TMP_MARK;

      ASSERT (dn >= 2);		/* to adhere to gpmpn_sbpi1_div_qr's limits */
      ASSERT (nn - dn >= 1);	/* to adhere to gpmpn_sbpi1_div_qr's limits */
      ASSERT (dp[0] & 1);

      tp = TMP_SALLOC_LIMBS (dn);

      qn = nn - dn;

      if (qn > dn)
      {
        /* Reduce qn mod dn without division, optimizing small operations.  */
        do
          qn -= dn;
        while (qn > dn);

        /* Perform the typically smaller block first.  */
        if (BELOW_THRESHOLD (qn, DC_BDIV_QR_THRESHOLD))
          cy = gpmpn_sbpi1_bdiv_qr (qp, np, 2 * qn, dp, qn, dinv);
        else
          cy = gpmpn_dcpi1_bdiv_qr_n (qp, np, dp, qn, dinv, tp);

        rr = 0;
        if (qn != dn)
        {
          if (qn > dn - qn)
            gpmpn_mul (tp, qp, qn, dp + qn, dn - qn);
          else
            gpmpn_mul (tp, dp + qn, dn - qn, qp, qn);
          gpmpn_incr_u (tp + qn, cy);

          rr = gpmpn_add (np + qn, np + qn, nn - qn, tp, dn);
          cy = 0;
        }

        np += qn;
        qp += qn;

        qn = nn - dn - qn;
        do
        {
          rr += gpmpn_add_1 (np + dn, np + dn, qn, cy);
          cy = gpmpn_dcpi1_bdiv_qr_n (qp, np, dp, dn, dinv, tp);
          qp += dn;
          np += dn;
          qn -= dn;
        }
        while (qn > 0);
        TMP_FREE;
        return rr + cy;
      }

      if (BELOW_THRESHOLD (qn, DC_BDIV_QR_THRESHOLD))
        cy = gpmpn_sbpi1_bdiv_qr (qp, np, 2 * qn, dp, qn, dinv);
      else
        cy = gpmpn_dcpi1_bdiv_qr_n (qp, np, dp, qn, dinv, tp);

      rr = 0;
      if (qn != dn)
      {
        if (qn > dn - qn)
          gpmpn_mul (tp, qp, qn, dp + qn, dn - qn);
        else
          gpmpn_mul (tp, dp + qn, dn - qn, qp, qn);
        gpmpn_incr_u (tp + qn, cy);

        rr = gpmpn_add (np + qn, np + qn, nn - qn, tp, dn);
        cy = 0;
      }

      TMP_FREE;
      return rr + cy;
    }

  }
}