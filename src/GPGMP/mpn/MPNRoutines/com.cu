/* gpmpn_com - complement an mpn.

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
#pragma once
#include "GPGMP/gpgmp-impl.cuh"

#ifdef __GPGMP_MPN(com)
#undef gpmpn_com
#endif

namespace gpgmp
{

  namespace mpnRoutines
  {

    ANYCALLER void gpmpn_com(mp_ptr result_ptr, mp_srcptr operand_ptr, mp_size_t size)
    {
      mp_limb_t operand_limb;
      do
      {
        operand_limb = *operand_ptr++;
        *result_ptr++ = ~operand_limb & GMP_NUMB_MASK;
      } while (--size != 0);
    }

  }
}
