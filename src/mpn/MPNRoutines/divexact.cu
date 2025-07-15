/* gpmpn_divexact(qp,np,nn,dp,dn,tp) -- Divide N = {np,nn} by D = {dp,dn} storing
   the result in Q = {qp,nn-dn+1} expecting no remainder.  Overlap allowed
   between Q and N; all other overlap disallowed.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2007, 2009, 2017 Free Software Foundation, Inc.

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


#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp {
  namespace mpnRoutines {

    HOSTONLY void gpmpn_divexact (mp_ptr qp, mp_srcptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn)
    {
      unsigned shift;
      mp_size_t qn;
      mp_ptr tp;
      TMP_DECL;

      ASSERT (dn > 0);
      ASSERT (nn >= dn);
      ASSERT (dp[dn-1] > 0);

      while (dp[0] == 0)
        {
          ASSERT (np[0] == 0);
          dp++;
          np++;
          dn--;
          nn--;
        }

      if (dn == 1)
        {
          MPN_DIVREM_OR_DIVEXACT_1 (qp, np, nn, dp[0]);
          return;
        }

      TMP_MARK;

      qn = nn + 1 - dn;
      count_trailing_zeros (shift, dp[0]);

      if (shift > 0)
        {
          mp_ptr wp;
          mp_size_t ss;
          ss = (dn > qn) ? qn + 1 : dn;

          tp = TMP_ALLOC_LIMBS (ss);
          gpmpn_rshift (tp, dp, ss, shift);
          dp = tp;

          /* Since we have excluded dn == 1, we have nn > qn, and we need
      to shift one limb beyond qn. */
          wp = TMP_ALLOC_LIMBS (qn + 1);
          gpmpn_rshift (wp, np, qn + 1, shift);
          np = wp;
        }

      if (dn > qn)
        dn = qn;

      tp = TMP_ALLOC_LIMBS (gpmpn_bdiv_q_itch (qn, dn));
      gpmpn_bdiv_q (qp, np, qn, dp, dn, tp);
      TMP_FREE;

      /* Since bdiv_q computes -N/D (mod B^{qn}), we must negate now. */
      gpmpn_neg (qp, qp, qn);
    }

  }
}