#pragma once

#include "GPGMP/gpgmp-impl.cuh"

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


		ANYCALLER mp_size_t gpmpn_div_q_itch_intermediary_maximum(mp_size_t maxNumberLimbsInEitherNumOrDenom)
		{
			return gpmpn_div_q_itch_intermediary(maxNumberLimbsInEitherNumOrDenom, 1);
		}
    }
}