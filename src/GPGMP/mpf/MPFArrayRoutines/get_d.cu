

#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER double
    gpmpf_get_d(mpf_array_idx src)
    {
      mp_size_t size, abs_size;
      long exp;

      size = MPF_ARRAY_SIZES(src.array)[src.idx];
      if (UNLIKELY(size == 0))
        return 0.0;

      abs_size = ABS(size);
      exp = (MPF_ARRAY_EXPONENTS(src.array)[src.idx] - abs_size) * GMP_NUMB_BITS;
      return gpgmp::mpnRoutines::gpmpn_get_d(MPF_ARRAY_DATA_AT_IDX(src.array, src.idx), abs_size, size, exp);
    }

  }
}