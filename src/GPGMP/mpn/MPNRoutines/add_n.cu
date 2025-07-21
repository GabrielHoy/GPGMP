/* gpmpn_add_n -- Add equal length limb vectors.  */
#pragma once
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp {

	namespace mpnRoutines {

		ANYCALLER mp_limb_t gpmpn_add_n (mp_ptr result_ptr, mp_srcptr operand1_ptr, mp_srcptr operand2_ptr, mp_size_t size)
		{
			mp_limb_t operand1_limb, operand2_limb, sum_limb, result_limb, carry, carry1, carry2;

			ASSERT (size >= 1);
			ASSERT (MPN_SAME_OR_INCR_P (result_ptr, operand1_ptr, size));
			ASSERT (MPN_SAME_OR_INCR_P (result_ptr, operand2_ptr, size));

			carry = 0;
			do
				{
				operand1_limb = *operand1_ptr++;
				operand2_limb = *operand2_ptr++;
				sum_limb = operand1_limb + operand2_limb;
				carry1 = sum_limb < operand1_limb;
				result_limb = sum_limb + carry;
				carry2 = result_limb < sum_limb;
				carry = carry1 | carry2;
				*result_ptr++ = result_limb;
				}
			while (--size != 0);

			return carry;
		}

	}
}
