#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* Notice the use of prec+1 ensures mpf_trunc is equivalent to mpf_set if u
       is already an integer.  */
    ANYCALLER void gpmpf_trunc(mpf_array_idx returnIn, mpf_array_idx truncate)
    {
      mp_ptr rp;
      mp_srcptr up;
      mp_size_t size, asize, prec;
      mp_exp_t exp;

      exp = MPF_ARRAY_EXPONENTS(truncate.array)[truncate.idx];
      size = MPF_ARRAY_SIZES(truncate.array)[truncate.idx];
      if (size == 0 || exp <= 0)
      {
        /* u is only a fraction */
        MPF_ARRAY_SIZES(returnIn.array)[returnIn.idx] = 0;
        MPF_ARRAY_EXPONENTS(returnIn.array)[returnIn.idx] = 0;
        return;
      }

      up = MPF_ARRAY_DATA_AT_IDX(truncate.array, truncate.idx);
      MPF_ARRAY_EXPONENTS(returnIn.array)[returnIn.idx] = exp;
      asize = ABS(size);
      up += asize;

      /* skip fraction part of u */
      asize = MIN(asize, exp);

      /* don't lose precision in the copy */
      prec = returnIn.array->userSpecifiedPrecisionLimbCount + 1;

      /* skip excess over target precision */
      asize = MIN(asize, prec);

      up -= asize;
      rp = MPF_ARRAY_DATA_AT_IDX(returnIn.array, returnIn.idx);
      MPF_ARRAY_SIZES(returnIn.array)[returnIn.idx] = (size >= 0 ? asize : -asize);
      if (rp != up)
        MPN_COPY_INCR(rp, up, asize);
    }

  }
}