#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER int
    gpmpf_integer_p(mpf_array_idx f) __GMP_NOTHROW
    {
      mp_srcptr fp;
      mp_exp_t exp;
      mp_size_t size;

      size = MPF_ARRAY_SIZES(f.array)[f.idx];
      exp = MPF_ARRAY_EXPONENTS(f.array)[f.idx];
      if (exp <= 0)
        return (size == 0); /* zero is an integer,
             others have only fraction limbs */
      size = ABS(size);

      /* Ignore zeroes at the low end of F.  */
      for (fp = MPF_ARRAY_DATA_AT_IDX(f.array, f.idx); *fp == 0; ++fp)
        --size;

      /* no fraction limbs */
      return size <= exp;
    }

  }
}