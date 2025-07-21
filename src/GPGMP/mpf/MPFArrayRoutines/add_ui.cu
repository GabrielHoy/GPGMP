#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
	namespace mpfArrayRoutines
	{

		ANYCALLER void gpmpf_add_ui(mpf_array_idx sum, mpf_array_idx u, unsigned long int v)
		{
			mp_srcptr up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
			mp_ptr sump = MPF_ARRAY_DATA_AT_IDX(sum.array, sum.idx);
			mp_size_t usize, sumsize;
			mp_size_t prec = sum.array->userSpecifiedPrecisionLimbCount;
			mp_exp_t uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];

			usize = MPF_ARRAY_SIZES(u.array)[u.idx];
			if (usize <= 0)
			{
				if (usize == 0)
				{
					gpmpf_set_ui(sum, v);
					return;
				}
				else
				{
					//instead of trying to construct a negative mpf_t copy of u like the mpf_t version does,
					//we'll just negate u, subtract v from it, then negate u again after we're done so it remains generally unchanged.
					gpmpf_neg(u, u);
					//u is now the equivalent of 'u_negated' in the below original code...
					gpmpf_sub_ui(sum, u, v);
					MPF_ARRAY_SIZES(sum.array)[sum.idx] = -(MPF_ARRAY_SIZES(sum.array)[sum.idx]);
					//...now that we've handled the subtraction and negation of the sum, the 'addition' is complete,
					//now all we need to do is negate u again to restore it to its original value.
					gpmpf_neg(u, u);

					/* original code for negation/subtraction
					__mpf_struct u_negated;
					u_negated._mp_size = -usize;
					u_negated._mp_exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
					u_negated._mp_d = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
					gpmpf_sub_ui(sum, &u_negated, v);
					MPF_ARRAY_SIZES(sum.array)[sum.idx] = -(MPF_ARRAY_SIZES(sum.array)[sum.idx]);
					*/

					return;
				}
			}

			if (v == 0)
			{
			sum_is_u:
				if ((u.array != sum.array) || (u.idx != sum.idx))
				{
					sumsize = MIN(usize, prec + 1);
					MPN_COPY(MPF_ARRAY_DATA_AT_IDX(sum.array, sum.idx), up + usize - sumsize, sumsize);
					MPF_ARRAY_SIZES(sum.array)[sum.idx] = sumsize;
					MPF_ARRAY_EXPONENTS(sum.array)[sum.idx] = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
				}
				return;
			}

			if (uexp > 0)
			{
				/* U >= 1.  */
				if (uexp > prec)
				{
					/* U >> V, V is not part of final result.  */
					goto sum_is_u;
				}
				else
				{
					/* U's "limb point" is somewhere between the first limb
					   and the PREC:th limb.
					   Both U and V are part of the final result.  */
					if (uexp > usize)
					{
						/*   uuuuuu0000. */
						/* +          v. */
						/* We begin with moving U to the top of SUM, to handle
					   samevar(U,SUM).  */
						MPN_COPY_DECR(sump + uexp - usize, up, usize);
						sump[0] = v;
						MPN_ZERO(sump + 1, uexp - usize - 1);
						MPF_ARRAY_SIZES(sum.array)[sum.idx] = uexp;
						MPF_ARRAY_EXPONENTS(sum.array)[sum.idx] = uexp;
					}
					else
					{
						/*   uuuuuu.uuuu */
						/* +      v.     */
						mp_limb_t cy_limb;
						if (usize > prec)
						{
							/* Ignore excess limbs in U.  */
							up += usize - prec;
							usize -= usize - prec; /* Eq. usize = prec */
						}
						if (sump != up)
							MPN_COPY_INCR(sump, up, usize - uexp);
						cy_limb = gpgmp::mpnRoutines::gpmpn_add_1(sump + usize - uexp, up + usize - uexp,
											uexp, (mp_limb_t)v);
						sump[usize] = cy_limb;
						MPF_ARRAY_SIZES(sum.array)[sum.idx] = usize + cy_limb;
						MPF_ARRAY_EXPONENTS(sum.array)[sum.idx] = uexp + cy_limb;
					}
				}
			}
			else
			{
				/* U < 1, so V > U for sure.  */
				/* v.         */
				/*  .0000uuuu */
				if ((-uexp) >= prec)
				{
					sump[0] = v;
					MPF_ARRAY_SIZES(sum.array)[sum.idx] = 1;
					MPF_ARRAY_EXPONENTS(sum.array)[sum.idx] = 1;
				}
				else
				{
					if (usize + (-uexp) + 1 > prec)
					{
						/* Ignore excess limbs in U.  */
						up += usize + (-uexp) + 1 - prec;
						usize -= usize + (-uexp) + 1 - prec;
					}
					if (sump != up)
						MPN_COPY_INCR(sump, up, usize);
					MPN_ZERO(sump + usize, -uexp);
					sump[usize + (-uexp)] = v;
					MPF_ARRAY_SIZES(sum.array)[sum.idx] = usize + (-uexp) + 1;
					MPF_ARRAY_EXPONENTS(sum.array)[sum.idx] = 1;
				}
			}
		}
	}
}