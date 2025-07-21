/* gpmpn_cnd_add_n -- Compute R = U + V if CND != 0 or R = U if CND == 0.
   Both cases should take the same time and perform the exact same memory
   accesses, since this function is intended to be used where side-channel
   attack resilience is relevant.

Copyright 1992-1994, 1996, 2000, 2002, 2008, 2009, 2011, 2013 Free Software
Foundation, Inc.

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

#pragma once
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp {

	namespace mpnRoutines {

		ANYCALLER mp_limb_t gpmpn_cnd_add_n (mp_limb_t condition, mp_ptr return_ptr, mp_srcptr operand1_ptr, mp_srcptr operand2_ptr, mp_size_t size)
    {
      mp_limb_t operand1_limb, operand2_limb, sum_limb, result_limb, carry, carry1, carry2, mask;

      ASSERT (size >= 1);
      ASSERT (MPN_SAME_OR_SEPARATE_P (return_ptr, operand1_ptr, size));
      ASSERT (MPN_SAME_OR_SEPARATE_P (return_ptr, operand2_ptr, size));

      mask = -(mp_limb_t) (condition != 0);
      carry = 0;
      do
        {
          operand1_limb = *operand1_ptr++;
          operand2_limb = *operand2_ptr++ & mask;

          sum_limb = operand1_limb + operand2_limb;
          carry1 = sum_limb < operand1_limb;
          result_limb = sum_limb + carry;
          carry2 = result_limb < sum_limb;
          carry = carry1 | carry2;
          *return_ptr++ = result_limb;
        }
      while (--size != 0);

      return carry;
    }

  }
}