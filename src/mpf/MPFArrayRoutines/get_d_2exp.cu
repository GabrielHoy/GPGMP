

#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER double
    gpmpf_get_d_2exp(signed long int *expptr, mpf_array_idx src)
    {
      mp_size_t size, abs_size;
      mp_srcptr ptr;
      int cnt;

      size = MPF_ARRAY_SIZES(src.array)[src.idx];
      if (UNLIKELY(size == 0))
      {
        *expptr = 0;
        return 0.0;
      }

      ptr = MPF_ARRAY_DATA_AT_IDX(src.array, src.idx);
      abs_size = ABS(size);
      count_leading_zeros(cnt, ptr[abs_size - 1]);
      cnt -= GMP_NAIL_BITS;

      *expptr = MPF_ARRAY_EXPONENTS(src.array)[src.idx] * GMP_NUMB_BITS - cnt;
      return gpgmp::mpnRoutines::gpmpn_get_d(ptr, abs_size, size, -(abs_size * GMP_NUMB_BITS - cnt));
    }

  }
}