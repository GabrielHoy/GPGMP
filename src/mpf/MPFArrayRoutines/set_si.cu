#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void
    gpmpf_set_si(mpf_array_idx toSet, long val)
    {
      mp_size_t size;
      mp_limb_t vl;

      vl = (mp_limb_t)ABS_CAST(unsigned long int, val);

      MPF_ARRAY_DATA_AT_IDX(toSet.array, toSet.idx)[0] = vl & GMP_NUMB_MASK;
      size = vl != 0;

#if BITS_PER_ULONG > GMP_NUMB_BITS
      vl >>= GMP_NUMB_BITS;
      MPF_ARRAY_DATA_AT_IDX(toSet.array, toSet.idx)[1] = vl;
      size += (vl != 0);
#endif

      MPF_ARRAY_EXPONENTS(toSet.array)[toSet.idx] = size;
      MPF_ARRAY_SIZES(toSet.array)[toSet.idx] = val >= 0 ? size : -size;
    }

  }
}