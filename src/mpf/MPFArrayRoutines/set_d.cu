

#include "config.cuh"

#if HAVE_FLOAT_H
#include <float.h> /* for DBL_MAX */
#endif

#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_set_d(mpf_array_idx toSet, double valueSetTo)
    {
      int negative;

      DOUBLE_NAN_INF_ACTION(valueSetTo,
                            __gmp_invalid_operation(),
                            __gmp_invalid_operation());

      if (UNLIKELY(valueSetTo == 0))
      {
        MPF_ARRAY_SIZES(toSet.array)[toSet.idx] = 0;
        MPF_ARRAY_EXPONENTS(toSet.array)[toSet.idx] = 0;
        return;
      }
      negative = valueSetTo < 0;
      valueSetTo = ABS(valueSetTo);

      MPF_ARRAY_SIZES(toSet.array)[toSet.idx] = negative ? -LIMBS_PER_DOUBLE : LIMBS_PER_DOUBLE;
      MPF_ARRAY_EXPONENTS(toSet.array)[toSet.idx] = __gpgmp_extract_double(MPF_ARRAY_DATA_AT_IDX(toSet.array, toSet.idx), valueSetTo);
    }

  }
}