/* mpf_sub_ui -- Subtract an unsigned integer from a float.

Copyright 1993, 1994, 1996, 2001 Free Software Foundation, Inc.

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

namespace gpgmp
{
  namespace mpfRoutines
  {

    ANYCALLER void
    gpmpf_sub_ui(mpf_ptr sum, mpf_srcptr u, unsigned long int v, mp_limb_t* scratchSpace)
    {
      __mpf_struct vv;
      mp_limb_t vl;

      if (v == 0)
      {
        gpmpf_set(sum, u);
        return;
      }

      vl = v;
      vv._mp_size = 1;
      vv._mp_d = &vl;
      vv._mp_exp = 1;
      gpmpf_sub(sum, u, &vv, scratchSpace);
    }

  }
}