/* mpf_init_set_str -- Initialize a float and assign it from a string.

Copyright 1995, 1996, 2000, 2001, 2004 Free Software Foundation, Inc.

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

namespace gpgmp
{
  namespace mpfRoutines
  {

    ANYCALLER int
    gpmpf_init_set_str(mpf_ptr r, const char *s, int base)
    {
      mp_size_t prec = __gmp_default_fp_limb_precision;
      r->_mp_size = 0;
      r->_mp_exp = 0;
      r->_mp_prec = prec;
      r->_mp_d = __GMP_ALLOCATE_FUNC_LIMBS(prec + 1);

      return gpmpf_set_str(r, s, base);
    }

  }
}