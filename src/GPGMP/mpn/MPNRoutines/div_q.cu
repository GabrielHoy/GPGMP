/* gpmpn_div_q -- division for arbitrary size operands.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2009, 2010, 2015, 2018 Free Software Foundation, Inc.

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

		//This is for the SECOND scratch space parameter to gpmpn_div_q, which is used for intermediary operations.
        //The previous parameters to gpmpn_div_q remain the same as GMP.
        //This routine can be optimized further by investigating the return patterns of the _q_itch functions, but this wont be called in a hotpath so to save time this works.
		//A possibly helpful note - through experimental testing up to 10,000 numerator limbs, the output of this function is always maximized as the denominatorNumLimbs is lowered; at 1 limb there will always be enough space for any intermediary operations on numeratorNumLimbs, assuming denominatorNumLimbs <= numeratorNumLimbs.
        ANYCALLER mp_size_t gpmpn_div_q_itch_intermediary(mp_size_t numeratorNumLimbs, mp_size_t denominatorNumLimbs)
		{
			mp_size_t totalScratchNeeded = 0;

			//We shouldn't boil this down to a single MAX() macro like some other itch's do since it'd be calling these functions twice iirc,
			//so store the results of the function calls ahead of time to avoid wasting CPU time and build up to the final MAX() call
			mp_size_t quotientNumLimbs = numeratorNumLimbs - denominatorNumLimbs + 1;

			mp_size_t muDivQItch1 = gpmpn_mu_div_q_itch(numeratorNumLimbs + 1, denominatorNumLimbs, 0);
			mp_size_t muDivQItch2 = gpmpn_mu_div_q_itch(numeratorNumLimbs, denominatorNumLimbs, 0);

			mp_size_t mpDivApprQItch1 = gpmpn_mu_divappr_q_itch((2 * quotientNumLimbs + 1) + 1, quotientNumLimbs + 1, 0);
			mp_size_t mpDivApprQItch2 = gpmpn_mu_divappr_q_itch((2 * quotientNumLimbs + 1), quotientNumLimbs + 1, 0);

			mp_size_t firstDivBranchPossibility = ((totalScratchNeeded + denominatorNumLimbs) + MAX(muDivQItch1, muDivQItch2));
			mp_size_t secondDivBranchPossibility = ((quotientNumLimbs + 1) + ((2 * quotientNumLimbs + 1) + 1) + (MAX(((quotientNumLimbs + 1) + mpDivApprQItch1), (mpDivApprQItch2))) + (denominatorNumLimbs + quotientNumLimbs));

			totalScratchNeeded += MAX(firstDivBranchPossibility,secondDivBranchPossibility);

			//For the possible call to gpmpn_dcpi1_div_q_itch_maximum
			totalScratchNeeded += gpmpn_dcpi1_div_q_itch(numeratorNumLimbs, denominatorNumLimbs);

			return totalScratchNeeded;
		}

		// This serves as a direct wrapper for the MPN_COPY macro.
		// This is done to avoid CUDA kernel launch errors when this macro is used inline in GPGMP GPU routines.
		ANYCALLER void perform_MPN_COPY(mp_ptr a, mp_srcptr b, mp_size_t c)
		{
			MPN_COPY(a, b + 1, c);
		}

		// This serves as a direct wrapper for the gpmpn_cmp function.
		// This is done to avoid CUDA kernel launch errors when this function is used inline in GPGMP GPU routines.
		ANYCALLER int perform_gpmpn_cmp(mp_srcptr a, mp_srcptr b, mp_size_t c)
		{
			return gpmpn_cmp(a, b, c);
		}

/* Compute Q = N/D with truncation.
	N = {np,numeratorNumLimbs}
	D = {dp,denominatorNumLimbs}
	Q = {quotientStoreIn,numeratorNumLimbs-denominatorNumLimbs+1}
	T = {scratch,numeratorNumLimbs+1} is scratch space
N and D are both untouched by the computation.
N and T may overlap; pass the same space if N is irrelevant after the call,
but note that tp needs an extra limb.

Operand requirements:
	N >= D > 0
	dp[denominatorNumLimbs-1] != 0
	No overlap between the N, D, and Q areas.

This division function does not clobber its input operands, since it is
intended to support average-O(qn) division, and for that to be effective, it
cannot put requirements on callers to copy a O(numeratorNumLimbs) operand.

If a caller does not care about the value of {np,numeratorNumLimbs+1} after calling this
function, it should pass np also for the scratch argument.  This function
will then save some time and space by avoiding allocation and copying.
(FIXME: Is this a good design?  We only really save any copying for
already-normalised divisors, which should be rare.  It also prevents us from
reasonably asking for all scratch space we need.)

We write numeratorNumLimbs-denominatorNumLimbs+1 limbs for the quotient, but return void.  Why not return
the most significant quotient limb?  Look at the 4 main code blocks below
(consisting of an outer if-else where each arm contains an if-else). It is
tricky for the first code block, since the gpmpn_*_div_q calls will typically
generate all numeratorNumLimbs-denominatorNumLimbs+1 and return 0 or 1.  I don't see how to fix that unless
we generate the most significant quotient limb here, before calling
gpmpn_*_div_q, or put the quotient in a temporary area.  Since this is a
critical division case (the SB sub-case in particular) copying is not a good
idea.

It might make sense to split the if-else parts of the (qn + FUDGE
>= denominatorNumLimbs) blocks into separate functions, since we could promise quite
different things to callers in these two cases.  The 'then' case
benefits from np=scratch, and it could perhaps even tolerate quotientStoreIn=np,
saving some headache for many callers.

FIXME: Scratch allocation leaves a lot to be desired.  E.g., for the MU size
operands, we do not reuse the huge scratch for adjustments.  This can be a
serious waste of memory for the largest operands.
*/

/* FUDGE determines when to try getting an approximate quotient from the upper
parts of the dividend and divisor, then adjust.  N.B. FUDGE must be >= 2
for the code to be correct.  */
#define FUDGE 5 /* FIXME: tune this */

#define DC_DIV_Q_THRESHOLD DC_DIVAPPR_Q_THRESHOLD
#define MU_DIV_Q_THRESHOLD MU_DIVAPPR_Q_THRESHOLD
#define MUPI_DIV_Q_THRESHOLD MUPI_DIVAPPR_Q_THRESHOLD
#ifndef MUPI_DIVAPPR_Q_THRESHOLD
#define MUPI_DIVAPPR_Q_THRESHOLD MUPI_DIV_QR_THRESHOLD
#endif

		ANYCALLER void gpmpn_div_q(mp_ptr quotientStoreIn, mp_srcptr numeratorPtr, mp_size_t numeratorNumLimbs, mp_srcptr denominatorPtr, mp_size_t denominatorNumLimbs, mp_ptr scratch, mp_ptr intermediaryScratch)
		{
			mp_ptr new_denominatorPtr, new_numeratorPtr, scratch2, remainderProduct;
			mp_limb_t carry, highDenominatorLimb, quotientHighLimb;
			mp_size_t shiftedNumeratorLimbNum, quotientNumLimbs;
			gmp_pi1_t inverseOfDenominator;
			int leadingZerosInQuotientHighLimb;

			ASSERT(numeratorNumLimbs >= denominatorNumLimbs);
			ASSERT(denominatorNumLimbs > 0);
			ASSERT(denominatorPtr[denominatorNumLimbs - 1] != 0);
			ASSERT(!MPN_OVERLAP_P(quotientStoreIn, numeratorNumLimbs - denominatorNumLimbs + 1, numeratorPtr, numeratorNumLimbs));
			ASSERT(!MPN_OVERLAP_P(quotientStoreIn, numeratorNumLimbs - denominatorNumLimbs + 1, denominatorPtr, denominatorNumLimbs));
			ASSERT(MPN_SAME_OR_SEPARATE_P(numeratorPtr, scratch, numeratorNumLimbs));

			ASSERT_ALWAYS(FUDGE >= 2);

			highDenominatorLimb = denominatorPtr[denominatorNumLimbs - 1];
			if (denominatorNumLimbs == 1)
			{
				gpmpn_divrem_1(quotientStoreIn, 0L, numeratorPtr, numeratorNumLimbs, highDenominatorLimb);
				return;
			}

			quotientNumLimbs = numeratorNumLimbs - denominatorNumLimbs + 1; /* Quotient size, high limb might be zero */

			if (quotientNumLimbs + FUDGE >= denominatorNumLimbs)
			{
				/* |________________________|
									|_______|  */
				new_numeratorPtr = scratch;

				if (LIKELY((highDenominatorLimb & GMP_NUMB_HIGHBIT) == 0))
				{
					count_leading_zeros(leadingZerosInQuotientHighLimb, highDenominatorLimb);

					carry = gpmpn_lshift(new_numeratorPtr, numeratorPtr, numeratorNumLimbs, leadingZerosInQuotientHighLimb);
					new_numeratorPtr[numeratorNumLimbs] = carry;
					shiftedNumeratorLimbNum = numeratorNumLimbs + (carry != 0);

					new_denominatorPtr = intermediaryScratch;
					intermediaryScratch += denominatorNumLimbs;

					gpmpn_lshift(new_denominatorPtr, denominatorPtr, denominatorNumLimbs, leadingZerosInQuotientHighLimb);

					if (denominatorNumLimbs == 2)
					{
						quotientHighLimb = gpmpn_divrem_2(quotientStoreIn, 0L, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr);
					}
					else if (BELOW_THRESHOLD(denominatorNumLimbs, DC_DIV_Q_THRESHOLD) ||
							 BELOW_THRESHOLD(shiftedNumeratorLimbNum - denominatorNumLimbs, DC_DIV_Q_THRESHOLD))
					{
						invert_pi1(inverseOfDenominator, new_denominatorPtr[denominatorNumLimbs - 1], new_denominatorPtr[denominatorNumLimbs - 2]);
						quotientHighLimb = gpmpn_sbpi1_div_q(quotientStoreIn, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, denominatorNumLimbs, inverseOfDenominator.inv32);
					}
					else if (BELOW_THRESHOLD(denominatorNumLimbs, MUPI_DIV_Q_THRESHOLD) ||					 /* fast condition */
							 BELOW_THRESHOLD(numeratorNumLimbs, 2 * MU_DIV_Q_THRESHOLD) ||					 /* fast condition */
							 (double)(2 * (MU_DIV_Q_THRESHOLD - MUPI_DIV_Q_THRESHOLD)) * denominatorNumLimbs /* slow... */
									 + (double)MUPI_DIV_Q_THRESHOLD * numeratorNumLimbs >
								 (double)denominatorNumLimbs * numeratorNumLimbs) /* ...condition */
					{
						invert_pi1(inverseOfDenominator, new_denominatorPtr[denominatorNumLimbs - 1], new_denominatorPtr[denominatorNumLimbs - 2]);
						mp_limb_t* moreScratch = intermediaryScratch;
						intermediaryScratch += gpmpn_dcpi1_div_q_itch(shiftedNumeratorLimbNum, denominatorNumLimbs);
						quotientHighLimb = gpmpn_dcpi1_div_q(quotientStoreIn, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, denominatorNumLimbs, &inverseOfDenominator, moreScratch);
					}
					else
					{
						mp_size_t itch = gpmpn_mu_div_q_itch(shiftedNumeratorLimbNum, denominatorNumLimbs, 0);
						mp_ptr moreScratch = intermediaryScratch;
						intermediaryScratch += itch;
						quotientHighLimb = gpmpn_mu_div_q(quotientStoreIn, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, denominatorNumLimbs, moreScratch);
					}
					if (carry == 0)
					{
						quotientStoreIn[quotientNumLimbs - 1] = quotientHighLimb;
					}
					else
					{
						ASSERT(quotientHighLimb == 0);
					}
				}
				else /* divisor is already normalised */
				{
					if (new_numeratorPtr != numeratorPtr)
					{
						MPN_COPY(new_numeratorPtr, numeratorPtr, numeratorNumLimbs);
					}

					if (denominatorNumLimbs == 2)
					{
						quotientHighLimb = gpmpn_divrem_2(quotientStoreIn, 0L, new_numeratorPtr, numeratorNumLimbs, denominatorPtr);
					}
					else if (BELOW_THRESHOLD(denominatorNumLimbs, DC_DIV_Q_THRESHOLD) ||
							 BELOW_THRESHOLD(numeratorNumLimbs - denominatorNumLimbs, DC_DIV_Q_THRESHOLD))
					{
						invert_pi1(inverseOfDenominator, highDenominatorLimb, denominatorPtr[denominatorNumLimbs - 2]);
						quotientHighLimb = gpmpn_sbpi1_div_q(quotientStoreIn, new_numeratorPtr, numeratorNumLimbs, denominatorPtr, denominatorNumLimbs, inverseOfDenominator.inv32);
					}
					else if (BELOW_THRESHOLD(denominatorNumLimbs, MUPI_DIV_Q_THRESHOLD) ||					 /* fast condition */
							 BELOW_THRESHOLD(numeratorNumLimbs, 2 * MU_DIV_Q_THRESHOLD) ||					 /* fast condition */
							 (double)(2 * (MU_DIV_Q_THRESHOLD - MUPI_DIV_Q_THRESHOLD)) * denominatorNumLimbs /* slow... */
									 + (double)MUPI_DIV_Q_THRESHOLD * numeratorNumLimbs >
								 (double)denominatorNumLimbs * numeratorNumLimbs) /* ...condition */
					{
						invert_pi1(inverseOfDenominator, highDenominatorLimb, denominatorPtr[denominatorNumLimbs - 2]);
						mp_limb_t* moreScratch = intermediaryScratch;
						intermediaryScratch += gpmpn_dcpi1_div_q_itch(numeratorNumLimbs, denominatorNumLimbs);
						quotientHighLimb = gpmpn_dcpi1_div_q(quotientStoreIn, new_numeratorPtr, numeratorNumLimbs, denominatorPtr, denominatorNumLimbs, &inverseOfDenominator, moreScratch);
					}
					else
					{
						mp_size_t itch = gpmpn_mu_div_q_itch(numeratorNumLimbs, denominatorNumLimbs, 0);
						mp_ptr moreScratch = intermediaryScratch;
						intermediaryScratch += itch;
						quotientHighLimb = gpmpn_mu_div_q(quotientStoreIn, numeratorPtr, numeratorNumLimbs, denominatorPtr, denominatorNumLimbs, moreScratch);
					}
					quotientStoreIn[numeratorNumLimbs - denominatorNumLimbs] = quotientHighLimb;
				}
			}
			else
			{
				/* |________________________|
							|_________________|  */
				scratch2 = intermediaryScratch;
				intermediaryScratch += (quotientNumLimbs + 1);

				new_numeratorPtr = scratch;
				shiftedNumeratorLimbNum = 2 * quotientNumLimbs + 1;
				if (new_numeratorPtr == numeratorPtr)
				{
					/* We need {np,nn} to remain untouched until the final adjustment, so
					we need to allocate separate space for new_numeratorPtr.  */
					new_numeratorPtr = intermediaryScratch;
					intermediaryScratch += (shiftedNumeratorLimbNum + 1);
				}

				if (LIKELY((highDenominatorLimb & GMP_NUMB_HIGHBIT) == 0))
				{
					count_leading_zeros(leadingZerosInQuotientHighLimb, highDenominatorLimb);

					carry = gpmpn_lshift(new_numeratorPtr, numeratorPtr + numeratorNumLimbs - shiftedNumeratorLimbNum, shiftedNumeratorLimbNum, leadingZerosInQuotientHighLimb);
					new_numeratorPtr[shiftedNumeratorLimbNum] = carry;

					shiftedNumeratorLimbNum += (carry != 0);

					new_denominatorPtr = intermediaryScratch;
					intermediaryScratch += (quotientNumLimbs + 1);
					gpmpn_lshift(new_denominatorPtr, denominatorPtr + denominatorNumLimbs - (quotientNumLimbs + 1), quotientNumLimbs + 1, leadingZerosInQuotientHighLimb);
					new_denominatorPtr[0] |= denominatorPtr[denominatorNumLimbs - (quotientNumLimbs + 1) - 1] >> (GMP_NUMB_BITS - leadingZerosInQuotientHighLimb);

					if (quotientNumLimbs + 1 == 2)
					{
						quotientHighLimb = gpmpn_divrem_2(scratch2, 0L, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr);
					}
					else if (BELOW_THRESHOLD(quotientNumLimbs, DC_DIVAPPR_Q_THRESHOLD - 1))
					{
						invert_pi1(inverseOfDenominator, new_denominatorPtr[quotientNumLimbs], new_denominatorPtr[quotientNumLimbs - 1]);
						quotientHighLimb = gpmpn_sbpi1_divappr_q(scratch2, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, quotientNumLimbs + 1, inverseOfDenominator.inv32);
					}
					else if (BELOW_THRESHOLD(quotientNumLimbs, MU_DIVAPPR_Q_THRESHOLD - 1))
					{
						invert_pi1(inverseOfDenominator, new_denominatorPtr[quotientNumLimbs], new_denominatorPtr[quotientNumLimbs - 1]);
						quotientHighLimb = gpmpn_dcpi1_divappr_q(scratch2, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, quotientNumLimbs + 1, &inverseOfDenominator);
					}
					else
					{
						mp_size_t itch = gpmpn_mu_divappr_q_itch(shiftedNumeratorLimbNum, quotientNumLimbs + 1, 0);
						mp_ptr moreScratch = intermediaryScratch;
						intermediaryScratch += itch;
						quotientHighLimb = gpmpn_mu_divappr_q(scratch2, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, quotientNumLimbs + 1, moreScratch);
					}

					if (carry == 0)
					{
						scratch2[quotientNumLimbs] = quotientHighLimb;
					}
					else if (UNLIKELY(quotientHighLimb != 0))
					{
						/* This happens only when the quotient is close to B^n and
						gpmpn_*_divappr_q returned B^n.  */
						mp_size_t i, n;
						n = shiftedNumeratorLimbNum - (quotientNumLimbs + 1);
						for (i = 0; i < n; i++)
							scratch2[i] = GMP_NUMB_MAX;
						quotientHighLimb = 0; /* currently ignored */
					}
				}
				else /* divisor is already normalised */
				{
					MPN_COPY(new_numeratorPtr, numeratorPtr + numeratorNumLimbs - shiftedNumeratorLimbNum, shiftedNumeratorLimbNum); /* pointless if MU will be used */

					new_denominatorPtr = (mp_ptr)denominatorPtr + denominatorNumLimbs - (quotientNumLimbs + 1);

					if (quotientNumLimbs == 2 - 1)
					{
						quotientHighLimb = gpmpn_divrem_2(scratch2, 0L, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr);
					}
					else if (BELOW_THRESHOLD(quotientNumLimbs, DC_DIVAPPR_Q_THRESHOLD - 1))
					{
						invert_pi1(inverseOfDenominator, highDenominatorLimb, new_denominatorPtr[quotientNumLimbs - 1]);
						quotientHighLimb = gpmpn_sbpi1_divappr_q(scratch2, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, quotientNumLimbs + 1, inverseOfDenominator.inv32);
					}
					else if (BELOW_THRESHOLD(quotientNumLimbs, MU_DIVAPPR_Q_THRESHOLD - 1))
					{
						invert_pi1(inverseOfDenominator, highDenominatorLimb, new_denominatorPtr[quotientNumLimbs - 1]);
						quotientHighLimb = gpmpn_dcpi1_divappr_q(scratch2, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, quotientNumLimbs + 1, &inverseOfDenominator);
					}
					else
					{
						mp_size_t itch = gpmpn_mu_divappr_q_itch(shiftedNumeratorLimbNum, quotientNumLimbs + 1, 0);
						mp_ptr moreScratch = intermediaryScratch;
						intermediaryScratch += itch;
						quotientHighLimb = gpmpn_mu_divappr_q(scratch2, new_numeratorPtr, shiftedNumeratorLimbNum, new_denominatorPtr, quotientNumLimbs + 1, moreScratch);
					}
					scratch2[quotientNumLimbs] = quotientHighLimb;
				}

				perform_MPN_COPY(quotientStoreIn, scratch2 + 1, quotientNumLimbs);
				if (scratch2[0] <= 4)
				{
					mp_size_t rn;

					remainderProduct = intermediaryScratch;
					intermediaryScratch += (denominatorNumLimbs + quotientNumLimbs);
					gpmpn_mul(remainderProduct, denominatorPtr, denominatorNumLimbs, scratch2 + 1, quotientNumLimbs);
					rn = denominatorNumLimbs + quotientNumLimbs;
					rn -= remainderProduct[rn - 1] == 0;

					int cmpResult = perform_gpmpn_cmp(numeratorPtr, remainderProduct, numeratorNumLimbs);
					if (rn > numeratorNumLimbs || cmpResult < 0)
					{
						MPN_DECR_U(quotientStoreIn, quotientNumLimbs, 1);
					}
					return;
				}
			}

			return;
		}

	}
}