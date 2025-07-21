

#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER mp_bitcnt_t
    gpmpf_get_prec(mpf_array_idx getFrom) __GMP_NOTHROW
    {
      return __GMPF_PREC_TO_BITS(getFrom.array->userSpecifiedPrecisionLimbCount);
    }

  }
}