#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER size_t gpmpf_size(mpf_array_idx getSizeOf) __GMP_NOTHROW
    {
      return ABS(MPF_ARRAY_SIZES(getSizeOf.array)[getSizeOf.idx]);
    }

  }
}