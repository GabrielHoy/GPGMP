/* gpmpn_add_n_sub_n -- Add and Subtract two limb vectors of equal, non-zero length.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1999-2001, 2006 Free Software Foundation, Inc.

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

#ifndef L1_CACHE_SIZE
#define L1_CACHE_SIZE 8192	/* only 68040 has less than this */ //TODO: Find out if this holds true for all "recent-ish" GPU architectures as well
#endif

#define PART_SIZE (L1_CACHE_SIZE / GMP_LIMB_BYTES / 6)

namespace gpgmp {

	namespace mpnRoutines {


    /* gpmpn_add_n_sub_n.
      r1[] = s1[] + s2[]
      r2[] = s1[] - s2[]
      All operands have n limbs.
      In-place operations allowed.  */
    ANYCALLER mp_limb_t gpmpn_add_n_sub_n (mp_ptr r1p, mp_ptr r2p, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n)
    {
      mp_limb_t acyn, acyo;		/* carry for add */
      mp_limb_t scyn, scyo;		/* carry for subtract */
      mp_size_t off;		/* offset in operands */
      mp_size_t this_n;		/* size of current chunk */

      /* We alternatingly add and subtract in chunks that fit into the (L1)
        cache.  Since the chunks are several hundred limbs, the function call
        overhead is insignificant, but we get much better locality.  */

      /* We have three variant of the inner loop, the proper loop is chosen
        depending on whether r1 or r2 are the same operand as s1 or s2.  */

      if (r1p != s1p && r1p != s2p)
      {
        /* r1 is not identical to either input operand.  We can therefore write
        to r1 directly, without using temporary storage.  */
        acyo = 0;
        scyo = 0;
        for (off = 0; off < n; off += PART_SIZE)
        {
          this_n = MIN (n - off, PART_SIZE);
          acyn = gpmpn_add_n (r1p + off, s1p + off, s2p + off, this_n);
          acyo = acyn + gpmpn_add_1 (r1p + off, r1p + off, this_n, acyo);
          scyn = gpmpn_sub_n (r2p + off, s1p + off, s2p + off, this_n);
          scyo = scyn + gpmpn_sub_1 (r2p + off, r2p + off, this_n, scyo);
        }
      }
      else if (r2p != s1p && r2p != s2p)
      {
        /* r2 is not identical to either input operand.  We can therefore write
        to r2 directly, without using temporary storage.  */
        acyo = 0;
        scyo = 0;
        for (off = 0; off < n; off += PART_SIZE)
        {
          this_n = MIN (n - off, PART_SIZE);
          scyn = gpmpn_sub_n (r2p + off, s1p + off, s2p + off, this_n);
          scyo = scyn + gpmpn_sub_1 (r2p + off, r2p + off, this_n, scyo);
          acyn = gpmpn_add_n (r1p + off, s1p + off, s2p + off, this_n);
          acyo = acyn + gpmpn_add_1 (r1p + off, r1p + off, this_n, acyo);
        }
      }
      else
      {
        /* r1 and r2 are identical to s1 and s2 (r1==s1 and r2==s2 or vice versa)
    Need temporary storage.  */
        mp_limb_t tp[PART_SIZE];
        acyo = 0;
        scyo = 0;
        for (off = 0; off < n; off += PART_SIZE)
        {
          this_n = MIN (n - off, PART_SIZE);
          acyn = gpmpn_add_n (tp, s1p + off, s2p + off, this_n);
          acyo = acyn + gpmpn_add_1 (tp, tp, this_n, acyo);
          scyn = gpmpn_sub_n (r2p + off, s1p + off, s2p + off, this_n);
          scyo = scyn + gpmpn_sub_1 (r2p + off, r2p + off, this_n, scyo);
          MPN_COPY (r1p + off, tp, this_n);
        }
      }

      return 2 * acyo + scyo;
    }

  }
}