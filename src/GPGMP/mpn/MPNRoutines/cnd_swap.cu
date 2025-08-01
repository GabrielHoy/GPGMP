/* gpmpn_cnd_swap

   Contributed to the GNU project by Niels Möller

Copyright 2013, 2015 Free Software Foundation, Inc.

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

    //Conditional swap of two mp_limb_t arrays using bitwise operations to protect against timing attacks.
		ANYCALLER void gpmpn_cnd_swap (mp_limb_t cnd, volatile mp_limb_t *ap, volatile mp_limb_t *bp, mp_size_t size)
    {
      volatile mp_limb_t mask = - (mp_limb_t) (cnd != 0);
      mp_size_t limbIdx;
      for (limbIdx = 0; limbIdx < size; limbIdx++)
        {
          mp_limb_t a, b, t;
          a = ap[limbIdx];
          b = bp[limbIdx];
          t = (a ^ b) & mask;
          ap[limbIdx] = a ^ t;
          bp[limbIdx] = b ^ t;
        }
    }

  }
}