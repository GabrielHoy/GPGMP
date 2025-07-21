

#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_set_ui(mpf_array_idx toSet, unsigned long val)
    {
      mp_size_t size;

      MPF_ARRAY_DATA_AT_IDX(toSet.array, toSet.idx)[0] = val & GMP_NUMB_MASK;
      size = val != 0;

#if BITS_PER_ULONG > GMP_NUMB_BITS
      val >>= GMP_NUMB_BITS;
      MPF_ARRAY_DATA_AT_IDX(toSet.array, toSet.idx)[1] = val;
      size += (val != 0);
#endif

      MPF_ARRAY_EXPONENTS(toSet.array)[toSet.idx] = MPF_ARRAY_SIZES(toSet.array)[toSet.idx] = size;
    }

  }
}