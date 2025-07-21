#include "gpgmp-impl.cuh"

namespace gpgmp
{
	namespace mpfArrayRoutines
	{
		ANYCALLER void
		gpmpf_sub(mpf_array_idx r, mpf_array_idx u, mpf_array_idx v)
		{
			MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_SUB);
			mp_srcptr up, vp;
			mp_ptr rp;
			mp_size_t usize, vsize, rsize;
			mp_size_t prec;
			mp_exp_t exp;
			mp_size_t ediff;
			mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);
			int negate;

			usize = MPF_ARRAY_SIZES(u.array)[u.idx];
			vsize = MPF_ARRAY_SIZES(v.array)[v.idx];

			/* Handle special cases that don't work in generic code below.  */
			if (usize == 0)
			{
				gpmpf_neg(r, v);
				return;
			}
			if (vsize == 0)
			{
				if ((r.idx != u.idx) || (r.array != u.array))
					gpmpf_set(r, u);
				return;
			}

			/* If signs of U and V are different, perform addition.  */
			if ((usize ^ vsize) < 0)
			{
				__mpf_struct v_negated;
				v_negated._mp_size = -vsize;
				v_negated._mp_exp = MPF_ARRAY_EXPONENTS(v.array)[v.idx];
				v_negated._mp_d = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
				gpgmp::internal::mpfArrayRoutines::gpmpf_add_mpf_t_to_array_idx(r, u, &v_negated);
				return;
			}

			/* Signs are now known to be the same.  */
			negate = usize < 0;

			/* Make U be the operand with the largest exponent.  */
			if (MPF_ARRAY_EXPONENTS(u.array)[u.idx] < MPF_ARRAY_EXPONENTS(v.array)[v.idx])
			{
				mpf_array_idx t = u;
				u = v;
				v = t;
				negate ^= 1;
				usize = MPF_ARRAY_SIZES(u.array)[u.idx];
				vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
			}

			usize = ABS(usize);
			vsize = ABS(vsize);
			up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
			vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
			rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
			prec = r.array->userSpecifiedPrecisionLimbCount + 1;
			exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
			ediff = exp - MPF_ARRAY_EXPONENTS(v.array)[v.idx];

			/* If ediff is 0 or 1, we might have a situation where the operands are
			   extremely close.  We need to scan the operands from the most significant
			   end ignore the initial parts that are equal.  */
			if (ediff <= 1)
			{
				if (ediff == 0)
				{
					/* Skip leading limbs in U and V that are equal.  */
					/* This loop normally exits immediately.  Optimize for that.  */
					while (up[usize - 1] == vp[vsize - 1])
					{
						usize--;
						vsize--;
						exp--;

						if (usize == 0)
						{
							/* u cancels high limbs of v, result is rest of v */
							negate ^= 1;
						cancellation:
							/* strip high zeros before truncating to prec */
							while (vsize != 0 && vp[vsize - 1] == 0)
							{
								vsize--;
								exp--;
							}
							if (vsize > prec)
							{
								vp += vsize - prec;
								vsize = prec;
							}
							MPN_COPY_INCR(rp, vp, vsize);
							rsize = vsize;
							goto done;
						}
						if (vsize == 0)
						{
							vp = up;
							vsize = usize;
							goto cancellation;
						}
					}

					if (up[usize - 1] < vp[vsize - 1])
					{
						/* For simplicity, swap U and V.  Note that since the loop above
					   wouldn't have exited unless up[usize - 1] and vp[vsize - 1]
					   were non-equal, this if-statement catches all cases where U
					   is smaller than V.  */

						MPN_SRCPTR_SWAP(up, usize, vp, vsize);
						negate ^= 1;
						/* negating ediff not necessary since it is 0.  */
					}

					/* Check for
					   x+1 00000000 ...
						x  ffffffff ... */
					if (up[usize - 1] != vp[vsize - 1] + 1)
						goto general_case;
					usize--;
					vsize--;
					exp--;
				}
				else /* ediff == 1 */
				{
					/* Check for
					   1 00000000 ...
					   0 ffffffff ... */

					if (up[usize - 1] != 1 || vp[vsize - 1] != GMP_NUMB_MAX || (usize >= 2 && up[usize - 2] != 0))
						goto general_case;

					usize--;
					exp--;
				}

				/* Skip sequences of 00000000/ffffffff */
				while (vsize != 0 && usize != 0 && up[usize - 1] == 0 && vp[vsize - 1] == GMP_NUMB_MAX)
				{
					usize--;
					vsize--;
					exp--;
				}

				if (usize == 0)
				{
					while (vsize != 0 && vp[vsize - 1] == GMP_NUMB_MAX)
					{
						vsize--;
						exp--;
					}
				}
				else if (usize > prec - 1)
				{
					up += usize - (prec - 1);
					usize = prec - 1;
				}
				if (vsize > prec - 1)
				{
					vp += vsize - (prec - 1);
					vsize = prec - 1;
				}

				{
					mp_limb_t cy_limb;
					if (vsize == 0)
					{
						MPN_COPY(scratchSpace, up, usize);
						scratchSpace[usize] = 1;
						rsize = usize + 1;
						exp++;
						goto normalized;
					}
					if (usize == 0)
					{
						cy_limb = gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, vsize);
						rsize = vsize;
					}
					else if (usize >= vsize)
					{
						/* uuuu     */
						/* vv       */
						mp_size_t size;
						size = usize - vsize;
						MPN_COPY(scratchSpace, up, size);
						cy_limb = gpgmp::mpnRoutines::gpmpn_sub_n(scratchSpace + size, up + size, vp, vsize);
						rsize = usize;
					}
					else /* (usize < vsize) */
					{
						/* uuuu     */
						/* vvvvvvv  */
						mp_size_t size;
						size = vsize - usize;
						cy_limb = gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size);
						cy_limb = gpgmp::mpnRoutines::gpmpn_sub_nc(scratchSpace + size, up, vp + size, usize, cy_limb);
						rsize = vsize;
					}
					if (cy_limb == 0)
					{
						scratchSpace[rsize] = 1;
						rsize++;
						exp++;
						goto normalized;
					}
					goto normalize;
				}
			}

		general_case:
			/* If U extends beyond PREC, ignore the part that does.  */
			if (usize > prec)
			{
				up += usize - prec;
				usize = prec;
			}

			/* If V extends beyond PREC, ignore the part that does.
			   Note that this may make vsize negative.  */
			if (vsize + ediff > prec)
			{
				vp += vsize + ediff - prec;
				vsize = prec - ediff;
			}

			if (ediff >= prec)
			{
				/* V completely cancelled.  */
				if (rp != up)
					MPN_COPY(rp, up, usize);
				rsize = usize;
			}
			else
			{
				/* Allocate temp space for the result.  Allocate
			   just vsize + ediff later???  */

				/* Locate the least significant non-zero limb in (the needed
			   parts of) U and V, to simplify the code below.  */
				for (;;)
				{
					if (vsize == 0)
					{
						MPN_COPY(rp, up, usize);
						rsize = usize;
						goto done;
					}
					if (vp[0] != 0)
						break;
					vp++, vsize--;
				}
				for (;;)
				{
					if (usize == 0)
					{
						MPN_COPY(rp, vp, vsize);
						rsize = vsize;
						negate ^= 1;
						goto done;
					}
					if (up[0] != 0)
						break;
					up++, usize--;
				}

				/* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
				/* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

				if (usize > ediff)
				{
					/* U and V partially overlaps.  */
					if (ediff == 0)
					{
						/* Have to compare the leading limbs of u and v
					   to determine whether to compute u - v or v - u.  */
						if (usize >= vsize)
						{
							/* uuuu     */
							/* vv       */
							mp_size_t size;
							size = usize - vsize;
							MPN_COPY(scratchSpace, up, size);
							gpgmp::mpnRoutines::gpmpn_sub_n(scratchSpace + size, up + size, vp, vsize);
							rsize = usize;
						}
						else /* (usize < vsize) */
						{
							/* uuuu     */
							/* vvvvvvv  */
							mp_size_t size;
							size = vsize - usize;
							ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size));
							gpgmp::mpnRoutines::gpmpn_sub_nc(scratchSpace + size, up, vp + size, usize, CNST_LIMB(1));
							rsize = vsize;
						}
					}
					else
					{
						if (vsize + ediff <= usize)
						{
							/* uuuu     */
							/*   v      */
							mp_size_t size;
							size = usize - ediff - vsize;
							MPN_COPY(scratchSpace, up, size);
							gpgmp::mpnRoutines::gpmpn_sub(scratchSpace + size, up + size, usize - size, vp, vsize);
							rsize = usize;
						}
						else
						{
							/* uuuu     */
							/*   vvvvv  */
							mp_size_t size;
							rsize = vsize + ediff;
							size = rsize - usize;
							ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size));
							gpgmp::mpnRoutines::gpmpn_sub(scratchSpace + size, up, usize, vp + size, usize - ediff);
							/* Should we use sub_nc then sub_1? */
							MPN_DECR_U(scratchSpace + size, usize, CNST_LIMB(1));
						}
					}
				}
				else
				{
					/* uuuu     */
					/*      vv  */
					mp_size_t size, i;
					size = vsize + ediff - usize;
					ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, vsize));
					for (i = vsize; i < size; i++)
						scratchSpace[i] = GMP_NUMB_MAX;
					gpgmp::mpnRoutines::gpmpn_sub_1(scratchSpace + size, up, usize, (mp_limb_t)1);
					rsize = size + usize;
				}

			normalize:
				/* Full normalize.  Optimize later.  */
				while (rsize != 0 && scratchSpace[rsize - 1] == 0)
				{
					rsize--;
					exp--;
				}
			normalized:
				MPN_COPY(rp, scratchSpace, rsize);
			}

		done:
			if (rsize == 0)
			{
				MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
				MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
			}
			else
			{
				MPF_ARRAY_SIZES(r.array)[r.idx] = negate ? -rsize : rsize;
				MPF_ARRAY_EXPONENTS(r.array)[r.idx] = exp;
			}
		}

	}


	namespace internal
	{
		namespace mpfArrayRoutines
		{
			//Meant for internal use,
			//Equivalent to gpmpf_sub, but using an mpf_t subtrahend instead of another mpf_array_idx.
			ANYCALLER void
			gpmpf_sub_mpf_t_from_array_idx(mpf_array_idx r, mpf_array_idx u, mpf_srcptr v)
			{
				MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_SUB);
				mp_srcptr up, vp;
				mp_ptr rp;
				mp_size_t usize, vsize, rsize;
				mp_size_t prec;
				mp_exp_t exp;
				mp_size_t ediff;
				mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);
				int negate;

				usize = MPF_ARRAY_SIZES(u.array)[u.idx];
				vsize = SIZ(v);

				/* Handle special cases that don't work in generic code below.  */
				if (usize == 0)
				{
					gpgmp::internal::mpfArrayRoutines::gpmpf_neg_set_array_idx_from_mpf_t(r, v);
					return;
				}
				if (vsize == 0)
				{
					if ((r.array != u.array) || (r.idx != u.idx))
						gpgmp::mpfArrayRoutines::gpmpf_set(r, u);
					return;
				}

				/* If signs of U and V are different, perform addition.  */
				if ((usize ^ vsize) < 0)
				{
					__mpf_struct v_negated;
					v_negated._mp_size = -vsize;
					v_negated._mp_exp = EXP(v);
					v_negated._mp_d = PTR(v);
					gpgmp::internal::mpfArrayRoutines::gpmpf_add_mpf_t_to_array_idx(r, u, &v_negated);
					return;
				}

				/* Signs are now known to be the same.  */
				negate = usize < 0;

				bool utilizeEffectiveUV = false;
				mpf_srcptr effectiveU;
				mpf_t effectiveV;

				/* Make U be the operand with the largest exponent.  */
				if (MPF_ARRAY_EXPONENTS(u.array)[u.idx] < EXP(v))
				{
					utilizeEffectiveUV = true;
					effectiveU = v;
					effectiveV->_mp_exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
					effectiveV->_mp_size = MPF_ARRAY_SIZES(u.array)[u.idx];
					effectiveV->_mp_prec = u.array->userSpecifiedPrecisionLimbCount;
					effectiveV->_mp_d = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

					usize = effectiveU->_mp_size;
					vsize = effectiveV->_mp_size;
				}

				usize = ABS(usize);
				vsize = ABS(vsize);
				up = utilizeEffectiveUV ? PTR(effectiveU) : MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
				vp = PTR(utilizeEffectiveUV ? effectiveV : v);
				rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
				prec = r.array->userSpecifiedPrecisionLimbCount + 1;
				exp = utilizeEffectiveUV ? effectiveU->_mp_exp : MPF_ARRAY_EXPONENTS(u.array)[u.idx];
				ediff = exp - EXP(utilizeEffectiveUV ? effectiveV : v);

				/* If ediff is 0 or 1, we might have a situation where the operands are
				   extremely close.  We need to scan the operands from the most significant
				   end ignore the initial parts that are equal.  */
				if (ediff <= 1)
				{
					if (ediff == 0)
					{
						/* Skip leading limbs in U and V that are equal.  */
						/* This loop normally exits immediately.  Optimize for that.  */
						while (up[usize - 1] == vp[vsize - 1])
						{
							usize--;
							vsize--;
							exp--;

							if (usize == 0)
							{
								/* u cancels high limbs of v, result is rest of v */
								negate ^= 1;
							cancellation:
								/* strip high zeros before truncating to prec */
								while (vsize != 0 && vp[vsize - 1] == 0)
								{
									vsize--;
									exp--;
								}
								if (vsize > prec)
								{
									vp += vsize - prec;
									vsize = prec;
								}
								MPN_COPY_INCR(rp, vp, vsize);
								rsize = vsize;
								goto done;
							}
							if (vsize == 0)
							{
								vp = up;
								vsize = usize;
								goto cancellation;
							}
						}

						if (up[usize - 1] < vp[vsize - 1])
						{
							/* For simplicity, swap U and V.  Note that since the loop above
						   wouldn't have exited unless up[usize - 1] and vp[vsize - 1]
						   were non-equal, this if-statement catches all cases where U
						   is smaller than V.  */
							MPN_SRCPTR_SWAP(up, usize, vp, vsize);
							negate ^= 1;
							/* negating ediff not necessary since it is 0.  */
						}

						/* Check for
						   x+1 00000000 ...
							x  ffffffff ... */
						if (up[usize - 1] != vp[vsize - 1] + 1)
							goto general_case;
						usize--;
						vsize--;
						exp--;
					}
					else /* ediff == 1 */
					{
						/* Check for
						   1 00000000 ...
						   0 ffffffff ... */

						if (up[usize - 1] != 1 || vp[vsize - 1] != GMP_NUMB_MAX || (usize >= 2 && up[usize - 2] != 0))
							goto general_case;

						usize--;
						exp--;
					}

					/* Skip sequences of 00000000/ffffffff */
					while (vsize != 0 && usize != 0 && up[usize - 1] == 0 && vp[vsize - 1] == GMP_NUMB_MAX)
					{
						usize--;
						vsize--;
						exp--;
					}

					if (usize == 0)
					{
						while (vsize != 0 && vp[vsize - 1] == GMP_NUMB_MAX)
						{
							vsize--;
							exp--;
						}
					}
					else if (usize > prec - 1)
					{
						up += usize - (prec - 1);
						usize = prec - 1;
					}
					if (vsize > prec - 1)
					{
						vp += vsize - (prec - 1);
						vsize = prec - 1;
					}

					{
						mp_limb_t cy_limb;
						if (vsize == 0)
						{
							MPN_COPY(scratchSpace, up, usize);
							scratchSpace[usize] = 1;
							rsize = usize + 1;
							exp++;
							goto normalized;
						}
						if (usize == 0)
						{
							cy_limb = gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, vsize);
							rsize = vsize;
						}
						else if (usize >= vsize)
						{
							/* uuuu     */
							/* vv       */
							mp_size_t size;
							size = usize - vsize;
							MPN_COPY(scratchSpace, up, size);
							cy_limb = gpgmp::mpnRoutines::gpmpn_sub_n(scratchSpace + size, up + size, vp, vsize);
							rsize = usize;
						}
						else /* (usize < vsize) */
						{
							/* uuuu     */
							/* vvvvvvv  */
							mp_size_t size;
							size = vsize - usize;
							cy_limb = gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size);
							cy_limb = gpgmp::mpnRoutines::gpmpn_sub_nc(scratchSpace + size, up, vp + size, usize, cy_limb);
							rsize = vsize;
						}
						if (cy_limb == 0)
						{
							scratchSpace[rsize] = 1;
							rsize++;
							exp++;
							goto normalized;
						}
						goto normalize;
					}
				}

			general_case:
				/* If U extends beyond PREC, ignore the part that does.  */
				if (usize > prec)
				{
					up += usize - prec;
					usize = prec;
				}

				/* If V extends beyond PREC, ignore the part that does.
				   Note that this may make vsize negative.  */
				if (vsize + ediff > prec)
				{
					vp += vsize + ediff - prec;
					vsize = prec - ediff;
				}

				if (ediff >= prec)
				{
					/* V completely cancelled.  */
					if (rp != up)
						MPN_COPY(rp, up, usize);
					rsize = usize;
				}
				else
				{
					/* Allocate temp space for the result.  Allocate
				   just vsize + ediff later???  */

					/* Locate the least significant non-zero limb in (the needed
				   parts of) U and V, to simplify the code below.  */
					for (;;)
					{
						if (vsize == 0)
						{
							MPN_COPY(rp, up, usize);
							rsize = usize;
							goto done;
						}
						if (vp[0] != 0)
							break;
						vp++, vsize--;
					}
					for (;;)
					{
						if (usize == 0)
						{
							MPN_COPY(rp, vp, vsize);
							rsize = vsize;
							negate ^= 1;
							goto done;
						}
						if (up[0] != 0)
							break;
						up++, usize--;
					}

					/* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
					/* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

					if (usize > ediff)
					{
						/* U and V partially overlaps.  */
						if (ediff == 0)
						{
							/* Have to compare the leading limbs of u and v
						   to determine whether to compute u - v or v - u.  */
							if (usize >= vsize)
							{
								/* uuuu     */
								/* vv       */
								mp_size_t size;
								size = usize - vsize;
								MPN_COPY(scratchSpace, up, size);
								gpgmp::mpnRoutines::gpmpn_sub_n(scratchSpace + size, up + size, vp, vsize);
								rsize = usize;
							}
							else /* (usize < vsize) */
							{
								/* uuuu     */
								/* vvvvvvv  */
								mp_size_t size;
								size = vsize - usize;
								ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size));
								gpgmp::mpnRoutines::gpmpn_sub_nc(scratchSpace + size, up, vp + size, usize, CNST_LIMB(1));
								rsize = vsize;
							}
						}
						else
						{
							if (vsize + ediff <= usize)
							{
								/* uuuu     */
								/*   v      */
								mp_size_t size;
								size = usize - ediff - vsize;
								MPN_COPY(scratchSpace, up, size);
								gpgmp::mpnRoutines::gpmpn_sub(scratchSpace + size, up + size, usize - size, vp, vsize);
								rsize = usize;
							}
							else
							{
								/* uuuu     */
								/*   vvvvv  */
								mp_size_t size;
								rsize = vsize + ediff;
								size = rsize - usize;
								ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size));
								gpgmp::mpnRoutines::gpmpn_sub(scratchSpace + size, up, usize, vp + size, usize - ediff);
								/* Should we use sub_nc then sub_1? */
								MPN_DECR_U(scratchSpace + size, usize, CNST_LIMB(1));
							}
						}
					}
					else
					{
						/* uuuu     */
						/*      vv  */
						mp_size_t size, i;
						size = vsize + ediff - usize;
						ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, vsize));
						for (i = vsize; i < size; i++)
							scratchSpace[i] = GMP_NUMB_MAX;
						gpgmp::mpnRoutines::gpmpn_sub_1(scratchSpace + size, up, usize, (mp_limb_t)1);
						rsize = size + usize;
					}

				normalize:
					/* Full normalize.  Optimize later.  */
					while (rsize != 0 && scratchSpace[rsize - 1] == 0)
					{
						rsize--;
						exp--;
					}
				normalized:
					MPN_COPY(rp, scratchSpace, rsize);
				}

			done:
				if (rsize == 0)
				{
					MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
					MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
				}
				else
				{
					MPF_ARRAY_SIZES(r.array)[r.idx] = negate ? -rsize : rsize;
					MPF_ARRAY_EXPONENTS(r.array)[r.idx] = exp;
				}
			}
		}

		namespace mpfRoutines
		{
			ANYCALLER void gpmpf_sub_mpf_array_idx_from_mpf_array_idx(mpf_ptr r, mpf_array_idx u, mpf_array_idx v, mp_limb_t* scratchSpace)
			{
				mp_srcptr up, vp;
				mp_ptr rp;
				mp_size_t usize, vsize, rsize;
				mp_size_t prec;
				mp_exp_t exp;
				mp_size_t ediff;
				int negate;

				usize = MPF_ARRAY_SIZES(u.array)[u.idx];
				vsize = MPF_ARRAY_SIZES(v.array)[v.idx];

				/* Handle special cases that don't work in generic code below.  */
				if (usize == 0)
				{
					gpgmp::internal::mpfRoutines::gpmpf_neg_mpf_array_idx(r, v);
					return;
				}
				if (vsize == 0)
				{
					gpgmp::internal::mpfRoutines::gpmpf_set_mpf_t_to_array_idx(r, u);
					return;
				}

				/* If signs of U and V are different, perform addition.  */
				if ((usize ^ vsize) < 0)
				{
					//Normally in GMP we'd create an __mpf_struct to serve as a negated form of V;
					//but we can do things a bit more efficiently I think here; let's call mpf_neg on V instead, use V for our addition directly, then mpf_neg it again after we're done with our addition to ensure v remains unchanged altogether.
					//mpf_neg in this case should literally just be flipping mp_size when called with (v,v).

					gpgmp::mpfArrayRoutines::gpmpf_neg(v, v);
					gpgmp::internal::mpfRoutines::gpmpf_add_mpf_array_idx_to_mpf_array_idx(r, u, v, scratchSpace);
					gpgmp::mpfArrayRoutines::gpmpf_neg(v, v);
					return;

					//I'll leave the original code below(not converted to mpf_array_idx's) for reference in case my intuition proves incorrect...
					/*
					__mpf_struct v_negated;
					v_negated._mp_size = -vsize;
					v_negated._mp_exp = EXP(v);
					v_negated._mp_d = PTR(v);
					gpmpf_add(r, u, &v_negated, scratchSpace);
					return;*/
				}

				/* Signs are now known to be the same.  */
				negate = usize < 0;

				/* Make U be the operand with the largest exponent.  */
				if (MPF_ARRAY_EXPONENTS(u.array)[u.idx] < MPF_ARRAY_EXPONENTS(v.array)[v.idx])
				{
					mpf_array_idx t;
					t = u;
					u = v;
					v = t;
					negate ^= 1;
					usize = MPF_ARRAY_SIZES(u.array)[u.idx];
					vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
				}

				usize = ABS(usize);
				vsize = ABS(vsize);
				up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
				vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
				rp = PTR(r);
				prec = PREC(r) + 1;
				exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
				ediff = exp - MPF_ARRAY_EXPONENTS(v.array)[v.idx];

				/* If ediff is 0 or 1, we might have a situation where the operands are
				extremely close.  We need to scan the operands from the most significant
				end ignore the initial parts that are equal.  */
				if (ediff <= 1)
				{
					if (ediff == 0)
					{
						/* Skip leading limbs in U and V that are equal.  */
						/* This loop normally exits immediately.  Optimize for that.  */
						while (up[usize - 1] == vp[vsize - 1])
						{
							usize--;
							vsize--;
							exp--;

							if (usize == 0)
							{
								/* u cancels high limbs of v, result is rest of v */
								negate ^= 1;
							cancellation:
								/* strip high zeros before truncating to prec */
								while (vsize != 0 && vp[vsize - 1] == 0)
								{
									vsize--;
									exp--;
								}
								if (vsize > prec)
								{
									vp += vsize - prec;
									vsize = prec;
								}
								MPN_COPY_INCR(rp, vp, vsize);
								rsize = vsize;
								goto done;
							}
							if (vsize == 0)
							{
								vp = up;
								vsize = usize;
								goto cancellation;
							}
						}

						if (up[usize - 1] < vp[vsize - 1])
						{
							/* For simplicity, swap U and V.  Note that since the loop above
						wouldn't have exited unless up[usize - 1] and vp[vsize - 1]
						were non-equal, this if-statement catches all cases where U
						is smaller than V.  */
							MPN_SRCPTR_SWAP(up, usize, vp, vsize);
							negate ^= 1;
							/* negating ediff not necessary since it is 0.  */
						}

						/* Check for
						x+1 00000000 ...
							x  ffffffff ... */
						if (up[usize - 1] != vp[vsize - 1] + 1)
							goto general_case;
						usize--;
						vsize--;
						exp--;
					}
					else /* ediff == 1 */
					{
						/* Check for
						1 00000000 ...
						0 ffffffff ... */

						if (up[usize - 1] != 1 || vp[vsize - 1] != GMP_NUMB_MAX || (usize >= 2 && up[usize - 2] != 0))
							goto general_case;

						usize--;
						exp--;
					}

					/* Skip sequences of 00000000/ffffffff */
					while (vsize != 0 && usize != 0 && up[usize - 1] == 0 && vp[vsize - 1] == GMP_NUMB_MAX)
					{
						usize--;
						vsize--;
						exp--;
					}

					if (usize == 0)
					{
						while (vsize != 0 && vp[vsize - 1] == GMP_NUMB_MAX)
						{
							vsize--;
							exp--;
						}
					}
					else if (usize > prec - 1)
					{
						up += usize - (prec - 1);
						usize = prec - 1;
					}
					if (vsize > prec - 1)
					{
						vp += vsize - (prec - 1);
						vsize = prec - 1;
					}

					{
						mp_limb_t cy_limb;
						if (vsize == 0)
						{
							MPN_COPY(scratchSpace, up, usize);
							scratchSpace[usize] = 1;
							rsize = usize + 1;
							exp++;
							goto normalized;
						}
						if (usize == 0)
						{
							cy_limb = gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, vsize);
							rsize = vsize;
						}
						else if (usize >= vsize)
						{
							/* uuuu     */
							/* vv       */
							mp_size_t size;
							size = usize - vsize;
							MPN_COPY(scratchSpace, up, size);
							cy_limb = gpgmp::mpnRoutines::gpmpn_sub_n(scratchSpace + size, up + size, vp, vsize);
							rsize = usize;
						}
						else /* (usize < vsize) */
						{
							/* uuuu     */
							/* vvvvvvv  */
							mp_size_t size;
							size = vsize - usize;
							cy_limb = gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size);
							cy_limb = gpgmp::mpnRoutines::gpmpn_sub_nc(scratchSpace + size, up, vp + size, usize, cy_limb);
							rsize = vsize;
						}
						if (cy_limb == 0)
						{
							scratchSpace[rsize] = 1;
							rsize++;
							exp++;
							goto normalized;
						}
						goto normalize;
					}
				}

			general_case:
				/* If U extends beyond PREC, ignore the part that does.  */
				if (usize > prec)
				{
					up += usize - prec;
					usize = prec;
				}

				/* If V extends beyond PREC, ignore the part that does.
				Note that this may make vsize negative.  */
				if (vsize + ediff > prec)
				{
					vp += vsize + ediff - prec;
					vsize = prec - ediff;
				}

				if (ediff >= prec)
				{
					/* V completely cancelled.  */
					if (rp != up)
						MPN_COPY(rp, up, usize);
					rsize = usize;
				}
				else
				{
					/* Allocate temp space for the result.  Allocate
				just vsize + ediff later???  */

					/* Locate the least significant non-zero limb in (the needed
				parts of) U and V, to simplify the code below.  */
					for (;;)
					{
						if (vsize == 0)
						{
							MPN_COPY(rp, up, usize);
							rsize = usize;
							goto done;
						}
						if (vp[0] != 0)
							break;
						vp++, vsize--;
					}
					for (;;)
					{
						if (usize == 0)
						{
							MPN_COPY(rp, vp, vsize);
							rsize = vsize;
							negate ^= 1;
							goto done;
						}
						if (up[0] != 0)
							break;
						up++, usize--;
					}

					/* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
					/* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

					if (usize > ediff)
					{
						/* U and V partially overlaps.  */
						if (ediff == 0)
						{
							/* Have to compare the leading limbs of u and v
						to determine whether to compute u - v or v - u.  */
							if (usize >= vsize)
							{
								/* uuuu     */
								/* vv       */
								mp_size_t size;
								size = usize - vsize;
								MPN_COPY(scratchSpace, up, size);
								gpgmp::mpnRoutines::gpmpn_sub_n(scratchSpace + size, up + size, vp, vsize);
								rsize = usize;
							}
							else /* (usize < vsize) */
							{
								/* uuuu     */
								/* vvvvvvv  */
								mp_size_t size;
								size = vsize - usize;
								ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size));
								gpgmp::mpnRoutines::gpmpn_sub_nc(scratchSpace + size, up, vp + size, usize, CNST_LIMB(1));
								rsize = vsize;
							}
						}
						else
						{
							if (vsize + ediff <= usize)
							{
								/* uuuu     */
								/*   v      */
								mp_size_t size;
								size = usize - ediff - vsize;
								MPN_COPY(scratchSpace, up, size);
								gpgmp::mpnRoutines::gpmpn_sub(scratchSpace + size, up + size, usize - size, vp, vsize);
								rsize = usize;
							}
							else
							{
								/* uuuu     */
								/*   vvvvv  */
								mp_size_t size;
								rsize = vsize + ediff;
								size = rsize - usize;
								ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, size));
								gpgmp::mpnRoutines::gpmpn_sub(scratchSpace + size, up, usize, vp + size, usize - ediff);
								/* Should we use sub_nc then sub_1? */
								MPN_DECR_U(scratchSpace + size, usize, CNST_LIMB(1));
							}
						}
					}
					else
					{
						/* uuuu     */
						/*      vv  */
						mp_size_t size, i;
						size = vsize + ediff - usize;
						ASSERT_CARRY(gpgmp::mpnRoutines::gpmpn_neg(scratchSpace, vp, vsize));
						for (i = vsize; i < size; i++)
							scratchSpace[i] = GMP_NUMB_MAX;
						gpgmp::mpnRoutines::gpmpn_sub_1(scratchSpace + size, up, usize, (mp_limb_t)1);
						rsize = size + usize;
					}

				normalize:
					/* Full normalize.  Optimize later.  */
					while (rsize != 0 && scratchSpace[rsize - 1] == 0)
					{
						rsize--;
						exp--;
					}
				normalized:
					MPN_COPY(rp, scratchSpace, rsize);
				}

			done:
				if (rsize == 0)
				{
					SIZ(r) = 0;
					EXP(r) = 0;
				}
				else
				{
					SIZ(r) = negate ? -rsize : rsize;
					EXP(r) = exp;
				}
			}
		}
	}
}