#pragma once

#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
    namespace mpnRoutines
    {
		ANYCALLER static inline mp_size_t gpmpn_div_q_itch_intermediary_maximum(mp_size_t maxNumberLimbsInEitherNumOrDenom)
		{
			return gpmpn_div_q_itch_intermediary(maxNumberLimbsInEitherNumOrDenom, 1);
		}
    }
}