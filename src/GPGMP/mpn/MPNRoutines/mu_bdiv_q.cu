/* gpmpn_mu_bdiv_q(qp,np,nn,dp,dn,tp) -- Compute {np,nn} / {dp,dn} mod B^nn.
   storing the result in {qp,nn}.  Overlap allowed between Q and N; all other
   overlap disallowed.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2005-2007, 2009, 2010, 2017 Free Software Foundation, Inc.

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
   The idea of the algorithm used herein is to compute a smaller inverted value
   than used in the standard Barrett algorithm, and thus save time in the
   Newton iterations, and pay just a small price when using the inverted value
   for developing quotient bits.  This algorithm was presented at ICMS 2006.
*/

#include "GPGMP/gpgmp-impl.cuh"
#include "GPGMP/longlong.cuh"

namespace gpgmp
{
	namespace mpnRoutines
	{

		/* N = {np,nn}
		   D = {dp,dn}

		   Requirements: N >= D
				 D >= 1
				 D odd
				 dn >= 2
				 nn >= 2
				 scratch space as determined by gpmpn_mu_bdiv_q_itch(nn,dn).

		   Write quotient to Q = {qp,nn}.

		   FIXME: When iterating, perhaps do the small step before loop, not after.
		   FIXME: Try to avoid the scalar divisions when computing inverse size.
		   FIXME: Trim allocation for (qn > dn) case, 3*dn might be possible.  In
			  particular, when dn==in, tp and rp could use the same space.
		   FIXME: Trim final quotient calculation to qn limbs of precision.
		*/
		ANYCALLER static void gpmpn_mu_bdiv_q_old(mp_ptr qp, mp_srcptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn, mp_ptr scratch)
		{
			mp_size_t qn;
			mp_size_t in;
			int cy, c0;
			mp_size_t tn, wn;

			qn = nn;

			ASSERT(dn >= 2);
			ASSERT(qn >= 2);

			if (qn > dn)
			{
				mp_size_t b;

				/* |_______________________|   dividend
					  |________|   divisor  */

#define ip scratch							 /* in */
#define rp (scratch + in)					 /* dn or rest >= binvert_itch(in) */
#define tp (scratch + in + dn)				 /* dn+in or next_size(dn) */
#define scratch_out (scratch + in + dn + tn) /* mulmod_bnm1_itch(next_size(dn)) */

				/* Compute an inverse size that is a nice partition of the quotient.  */
				b = (qn - 1) / dn + 1; /* ceil(qn/dn), number of blocks */
				in = (qn - 1) / b + 1; /* ceil(qn/b) = ceil(qn / ceil(qn/dn)) */

				/* Some notes on allocation:

			   When in = dn, R dies when gpmpn_mullo returns, if in < dn the low in
			   limbs of R dies at that point.  We could save memory by letting T live
			   just under R, and let the upper part of T expand into R. These changes
			   should reduce itch to perhaps 3dn.
				 */

				gpmpn_binvert(ip, dp, in, rp);

				cy = 0;

				MPN_COPY(rp, np, dn);
				np += dn;
				gpmpn_mullo_n(qp, rp, ip, in);
				qn -= in;

				while (qn > in)
				{
					if (BELOW_THRESHOLD(in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
						gpmpn_mul(tp, dp, dn, qp, in); /* mulhi, need tp[dn+in-1...in] */
					else
					{
						tn = gpmpn_mulmod_bnm1_next_size(dn);
						gpmpn_mulmod_bnm1(tp, tn, dp, dn, qp, in, scratch_out);
						wn = dn + in - tn; /* number of wrapped limbs */
						if (wn > 0)
						{
							c0 = gpmpn_sub_n(tp + tn, tp, rp, wn);
							gpmpn_decr_u(tp + wn, c0);
						}
					}

					qp += in;
					if (dn != in)
					{
						/* Subtract tp[dn-1...in] from partial remainder.  */
						cy += gpmpn_sub_n(rp, rp + in, tp + in, dn - in);
						if (cy == 2)
						{
							gpmpn_incr_u(tp + dn, 1);
							cy = 1;
						}
					}
					/* Subtract tp[dn+in-1...dn] from dividend.  */
					cy = gpmpn_sub_nc(rp + dn - in, np, tp + dn, in, cy);
					np += in;
					gpmpn_mullo_n(qp, rp, ip, in);
					qn -= in;
				}

				/* Generate last qn limbs.
			   FIXME: It should be possible to limit precision here, since qn is
			   typically somewhat smaller than dn.  No big gains expected.  */

				if (BELOW_THRESHOLD(in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
					gpmpn_mul(tp, dp, dn, qp, in); /* mulhi, need tp[qn+in-1...in] */
				else
				{
					tn = gpmpn_mulmod_bnm1_next_size(dn);
					gpmpn_mulmod_bnm1(tp, tn, dp, dn, qp, in, scratch_out);
					wn = dn + in - tn; /* number of wrapped limbs */
					if (wn > 0)
					{
						c0 = gpmpn_sub_n(tp + tn, tp, rp, wn);
						gpmpn_decr_u(tp + wn, c0);
					}
				}

				qp += in;
				if (dn != in)
				{
					cy += gpmpn_sub_n(rp, rp + in, tp + in, dn - in);
					if (cy == 2)
					{
						gpmpn_incr_u(tp + dn, 1);
						cy = 1;
					}
				}

				gpmpn_sub_nc(rp + dn - in, np, tp + dn, qn - (dn - in), cy);
				gpmpn_mullo_n(qp, rp, ip, qn);

#undef ip
#undef rp
#undef tp
#undef scratch_out
			}
			else
			{
				/* |_______________________|   dividend
				  |________________|   divisor  */

#define ip scratch						/* in */
#define tp (scratch + in)				/* qn+in or next_size(qn) or rest >= binvert_itch(in) */
#define scratch_out (scratch + in + tn) /* mulmod_bnm1_itch(next_size(qn)) */

				/* Compute half-sized inverse.  */
				in = qn - (qn >> 1);

				gpmpn_binvert(ip, dp, in, tp);

				gpmpn_mullo_n(qp, np, ip, in); /* low `in' quotient limbs */

				if (BELOW_THRESHOLD(in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
					gpmpn_mul(tp, dp, qn, qp, in); /* mulhigh */
				else
				{
					tn = gpmpn_mulmod_bnm1_next_size(qn);
					gpmpn_mulmod_bnm1(tp, tn, dp, qn, qp, in, scratch_out);
					wn = qn + in - tn; /* number of wrapped limbs */
					if (wn > 0)
					{
						c0 = gpmpn_cmp(tp, np, wn) < 0;
						gpmpn_decr_u(tp + wn, c0);
					}
				}

				gpmpn_sub_n(tp, np + in, tp + in, qn - in);
				gpmpn_mullo_n(qp + in, tp, ip, qn - in); /* high qn-in quotient limbs */

#undef ip
#undef tp
#undef scratch_out
			}
		}

		ANYCALLER void gpmpn_mu_bdiv_q(mp_ptr qp,
					  mp_srcptr np, mp_size_t nn,
					  mp_srcptr dp, mp_size_t dn,
					  mp_ptr scratch)
		{
			gpmpn_mu_bdiv_q_old(qp, np, nn, dp, dn, scratch);
			gpmpn_neg(qp, qp, nn);
		}

		ANYCALLER mp_size_t gpmpn_mu_bdiv_q_itch(mp_size_t nn, mp_size_t dn)
		{
			mp_size_t qn, in, tn, itch_binvert, itch_out, itches;
			mp_size_t b;

			ASSERT_ALWAYS(DC_BDIV_Q_THRESHOLD < MU_BDIV_Q_THRESHOLD);

			qn = nn;

			if (qn > dn)
			{
				b = (qn - 1) / dn + 1; /* ceil(qn/dn), number of blocks */
				in = (qn - 1) / b + 1; /* ceil(qn/b) = ceil(qn / ceil(qn/dn)) */
				if (BELOW_THRESHOLD(in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
				{
					tn = dn + in;
					itch_out = 0;
				}
				else
				{
					tn = gpmpn_mulmod_bnm1_next_size(dn);
					itch_out = gpmpn_mulmod_bnm1_itch(tn, dn, in);
				}
				itches = dn + tn + itch_out;
			}
			else
			{
				in = qn - (qn >> 1);
				if (BELOW_THRESHOLD(in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
				{
					tn = qn + in;
					itch_out = 0;
				}
				else
				{
					tn = gpmpn_mulmod_bnm1_next_size(qn);
					itch_out = gpmpn_mulmod_bnm1_itch(tn, qn, in);
				}
				itches = tn + itch_out;
			}

			itch_binvert = gpmpn_binvert_itch(in);
			return in + MAX(itches, itch_binvert);
		}

	}
}