/* gpmpn_and_n, gpmpn_ior_n, etc -- mpn logical operations.

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

#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpnRoutines
  {
    ANYCALLER void __MPN(and_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_and_n(rp, up, vp, n);
    }

    ANYCALLER void __MPN(andn_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_andn_n(rp, up, vp, n);
    }

    ANYCALLER void __MPN(nand_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_nand_n(rp, up, vp, n);
    }

    ANYCALLER void __MPN(ior_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_ior_n(rp, up, vp, n);
    }

    ANYCALLER void __MPN(iorn_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_iorn_n(rp, up, vp, n);
    }

    ANYCALLER void __MPN(nior_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_nior_n(rp, up, vp, n);
    }

    ANYCALLER void __MPN(xor_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_xor_n(rp, up, vp, n);
    }

    ANYCALLER void __MPN(xnor_n)(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
    {
      gpmpn_xnor_n(rp, up, vp, n);
    }

  }
}