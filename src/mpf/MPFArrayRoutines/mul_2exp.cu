#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* Multiples of GMP_NUMB_BITS in exp simply mean an amount added to EXP(u)
       to set EXP(r).  The remainder exp%GMP_NUMB_BITS is then a left shift for
       the limb data.

       If exp%GMP_NUMB_BITS == 0 then there's no shifting, we effectively just
       do an mpz_set with changed EXP(r).  Like mpz_set we take prec+1 limbs in
       this case.  Although just prec would suffice, it's nice to have
       mpf_mul_2exp with exp==0 come out the same as mpz_set.

       When shifting we take up to prec many limbs from the input.  Our shift is
       cy = mpn_lshift (PTR(r), PTR(u)+k, size, ...), where k is the number of
       low limbs dropped from u, and the carry out is stored to PTR(r)[size].

       It may be noted that the low limb PTR(r)[0] doesn't incorporate bits from
       PTR(u)[k-1] (when k>=1 makes that limb available).  Taking just prec
       limbs from the input (with the high non-zero) is enough bits for the
       application requested precision, there's no need for extra work.

       If r==u the shift will have overlapping operands.  When k==0 (ie. when
       usize <= prec), the overlap is supported by lshift (ie. dst == src).

       But when r==u and k>=1 (ie. usize > prec), we would have an invalid
       overlap (ie. mpn_lshift (rp, rp+k, ...)).  In this case we must instead
       use mpn_rshift (PTR(r)+1, PTR(u)+k, size, NUMB-shift) with the carry out
       stored to PTR(r)[0].  An rshift by NUMB-shift bits like this gives
       identical data, it's just its overlap restrictions which differ.

       Enhancements:

       The way mpn_lshift is used means successive mpf_mul_2exp calls on the
       same operand will accumulate low zero limbs, until prec+1 limbs is
       reached.  This is wasteful for subsequent operations.  When abs_usize <=
       prec, we should test the low exp%GMP_NUMB_BITS many bits of PTR(u)[0],
       ie. those which would be shifted out by an mpn_rshift.  If they're zero
       then use that mpn_rshift.  */

    ANYCALLER void
    gpmpf_mul_2exp(mpf_array_idx r, mpf_array_idx u, mp_bitcnt_t exp)
    {
      mp_srcptr up;
      mp_ptr rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
      mp_size_t usize;
      mp_size_t abs_usize;
      mp_size_t prec = r.array->userSpecifiedPrecisionLimbCount;
      mp_exp_t uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];

      if (UNLIKELY(usize == 0))
      {
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      abs_usize = ABS(usize);
      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

      if (exp % GMP_NUMB_BITS == 0)
      {
        prec++; /* retain more precision here as we don't need
         to account for carry-out here */
        if (abs_usize > prec)
        {
          up += abs_usize - prec;
          abs_usize = prec;
        }
        if (rp != up)
          MPN_COPY_INCR(rp, up, abs_usize);
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = uexp + exp / GMP_NUMB_BITS;
      }
      else
      {
        mp_limb_t cy_limb;
        mp_size_t adj;
        if (abs_usize > prec)
        {
          up += abs_usize - prec;
          abs_usize = prec;
          /* Use mpn_rshift since mpn_lshift operates downwards, and we
             therefore would clobber part of U before using that part, in case
             R is the same variable as U.  */
          cy_limb = gpgmp::mpnRoutines::gpmpn_rshift(rp + 1, up, abs_usize,
                               GMP_NUMB_BITS - exp % GMP_NUMB_BITS);
          rp[0] = cy_limb;
          adj = rp[abs_usize] != 0;
        }
        else
        {
          cy_limb = gpgmp::mpnRoutines::gpmpn_lshift(rp, up, abs_usize, exp % GMP_NUMB_BITS);
          rp[abs_usize] = cy_limb;
          adj = cy_limb != 0;
        }

        abs_usize += adj;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = uexp + exp / GMP_NUMB_BITS + adj;
      }
      MPF_ARRAY_SIZES(r.array)[r.idx] = usize >= 0 ? abs_usize : -abs_usize;
    }

  }
}