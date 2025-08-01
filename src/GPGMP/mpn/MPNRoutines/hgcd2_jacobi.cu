/* hgcd2_jacobi.c

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1996, 1998, 2000-2004, 2008, 2011, 2020 Free Software
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

#include "GPGMP/gpgmp-impl.cuh"
#include "GPGMP/longlong.cuh"
#include "GPGMP/mpn/MPNRoutines/hgcd2-div.cuh"

namespace gpgmp {
  namespace mpnRoutines {


		#if GMP_NAIL_BITS != 0
		#error Nails not implemented
		#endif

		ANYCALLER int gpmpn_hgcd2_jacobi (mp_limb_t ah, mp_limb_t al, mp_limb_t bh, mp_limb_t bl, struct hgcd_matrix1 *M, unsigned *bitsp)
		{
		mp_limb_t u00, u01, u10, u11;
		unsigned bits = *bitsp;

		if (ah < 2 || bh < 2)
			return 0;

		if (ah > bh || (ah == bh && al > bl))
			{
			sub_ddmmss (ah, al, ah, al, bh, bl);
			if (ah < 2)
			return 0;

			u00 = u01 = u11 = 1;
			u10 = 0;
			bits = gpmpn_jacobi_update (bits, 1, 1);
			}
		else
			{
			sub_ddmmss (bh, bl, bh, bl, ah, al);
			if (bh < 2)
			return 0;

			u00 = u10 = u11 = 1;
			u01 = 0;
			bits = gpmpn_jacobi_update (bits, 0, 1);
			}

		if (ah < bh)
			goto subtract_a;

		for (;;)
			{
			ASSERT (ah >= bh);
			if (ah == bh)
			goto done;

			if (ah < (CNST_LIMB(1) << (GMP_LIMB_BITS / 2)))
			{
			ah = (ah << (GMP_LIMB_BITS / 2) ) + (al >> (GMP_LIMB_BITS / 2));
			bh = (bh << (GMP_LIMB_BITS / 2) ) + (bl >> (GMP_LIMB_BITS / 2));

			break;
			}

			/* Subtract a -= q b, and multiply M from the right by (1 q ; 0
			1), affecting the second column of M. */
			ASSERT (ah > bh);
			sub_ddmmss (ah, al, ah, al, bh, bl);

			if (ah < 2)
			goto done;

			if (ah <= bh)
			{
			/* Use q = 1 */
			u01 += u00;
			u11 += u10;
			bits = gpmpn_jacobi_update (bits, 1, 1);
			}
			else
			{
			mp_limb_t r[2];
			mp_limb_t q = div2 (r, ah, al, bh, bl);
			al = r[0]; ah = r[1];
			if (ah < 2)
				{
				/* A is too small, but q is correct. */
				u01 += q * u00;
				u11 += q * u10;
				bits = gpmpn_jacobi_update (bits, 1, q & 3);
				goto done;
				}
			q++;
			u01 += q * u00;
			u11 += q * u10;
			bits = gpmpn_jacobi_update (bits, 1, q & 3);
			}
			subtract_a:
			ASSERT (bh >= ah);
			if (ah == bh)
			goto done;

			if (bh < (CNST_LIMB(1) << (GMP_LIMB_BITS / 2)))
			{
			ah = (ah << (GMP_LIMB_BITS / 2) ) + (al >> (GMP_LIMB_BITS / 2));
			bh = (bh << (GMP_LIMB_BITS / 2) ) + (bl >> (GMP_LIMB_BITS / 2));

			goto subtract_a1;
			}

			/* Subtract b -= q a, and multiply M from the right by (1 0 ; q
			1), affecting the first column of M. */
			sub_ddmmss (bh, bl, bh, bl, ah, al);

			if (bh < 2)
			goto done;

			if (bh <= ah)
			{
			/* Use q = 1 */
			u00 += u01;
			u10 += u11;
			bits = gpmpn_jacobi_update (bits, 0, 1);
			}
			else
			{
			mp_limb_t r[2];
			mp_limb_t q = div2 (r, bh, bl, ah, al);
			bl = r[0]; bh = r[1];
			if (bh < 2)
				{
				/* B is too small, but q is correct. */
				u00 += q * u01;
				u10 += q * u11;
				bits = gpmpn_jacobi_update (bits, 0, q & 3);
				goto done;
				}
			q++;
			u00 += q * u01;
			u10 += q * u11;
			bits = gpmpn_jacobi_update (bits, 0, q & 3);
			}
			}

		/* NOTE: Since we discard the least significant half limb, we don't get a
			truly maximal M (corresponding to |a - b| < 2^{GMP_LIMB_BITS +1}). */
		/* Single precision loop */
		for (;;)
			{
			ASSERT (ah >= bh);

			ah -= bh;
			if (ah < (CNST_LIMB (1) << (GMP_LIMB_BITS / 2 + 1)))
			break;

			if (ah <= bh)
			{
			/* Use q = 1 */
			u01 += u00;
			u11 += u10;
			bits = gpmpn_jacobi_update (bits, 1, 1);
			}
			else
			{
			mp_double_limb_t rq = div1 (ah, bh);
			mp_limb_t q = rq.d1;
			ah = rq.d0;

			if (ah < (CNST_LIMB(1) << (GMP_LIMB_BITS / 2 + 1)))
				{
				/* A is too small, but q is correct. */
				u01 += q * u00;
				u11 += q * u10;
				bits = gpmpn_jacobi_update (bits, 1, q & 3);
				break;
				}
			q++;
			u01 += q * u00;
			u11 += q * u10;
			bits = gpmpn_jacobi_update (bits, 1, q & 3);
			}
			subtract_a1:
			ASSERT (bh >= ah);

			bh -= ah;
			if (bh < (CNST_LIMB (1) << (GMP_LIMB_BITS / 2 + 1)))
			break;

			if (bh <= ah)
			{
			/* Use q = 1 */
			u00 += u01;
			u10 += u11;
			bits = gpmpn_jacobi_update (bits, 0, 1);
			}
			else
			{
			mp_double_limb_t rq = div1 (bh, ah);
			mp_limb_t q = rq.d1;
			bh = rq.d0;

			if (bh < (CNST_LIMB(1) << (GMP_LIMB_BITS / 2 + 1)))
				{
				/* B is too small, but q is correct. */
				u00 += q * u01;
				u10 += q * u11;
				bits = gpmpn_jacobi_update (bits, 0, q & 3);
				break;
				}
			q++;
			u00 += q * u01;
			u10 += q * u11;
			bits = gpmpn_jacobi_update (bits, 0, q & 3);
			}
			}

		done:
		M->u[0][0] = u00; M->u[0][1] = u01;
		M->u[1][0] = u10; M->u[1][1] = u11;
		*bitsp = bits;

		return 1;
		}

  }
}