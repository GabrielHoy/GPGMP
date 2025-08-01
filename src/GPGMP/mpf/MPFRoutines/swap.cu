/* mpf_swap (U, V) -- Swap U and V.

Copyright 1997, 1998, 2000, 2001, 2013 Free Software Foundation, Inc.

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

    ANYCALLER void
    gpmpf_swap(mpf_ptr u, mpf_ptr v) __GMP_NOTHROW
    {
      mp_ptr tptr;
      mp_size_t tprec;
      mp_size_t tsiz;
      mp_exp_t texp;

      tprec = PREC(u);
      PREC(u) = PREC(v);
      PREC(v) = tprec;

      tsiz = SIZ(u);
      SIZ(u) = SIZ(v);
      SIZ(v) = tsiz;

      texp = EXP(u);
      EXP(u) = EXP(v);
      EXP(v) = texp;

      tptr = PTR(u);
      PTR(u) = PTR(v);
      PTR(v) = tptr;
    }

  }
}