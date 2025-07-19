#include "config.cuh"

#if HAVE_FLOAT_H
#include <float.h> /* for DBL_MAX */
#endif

#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER int gpmpf_cmp_d(mpf_array_idx f, double d)
    {
      mp_limb_t darray[LIMBS_PER_DOUBLE];
      mpf_t df;

      /* d=NaN has no sensible return value, so raise an exception.
         d=Inf or -Inf is always bigger than z.  */
      DOUBLE_NAN_INF_ACTION(d,
                            __gmp_invalid_operation(),
                            return (d < 0.0 ? 1 : -1));

      if (d == 0.0)
        return MPF_ARRAY_SIZES(f.array)[f.idx];

      PTR(df) = darray;
      SIZ(df) = (d >= 0.0 ? LIMBS_PER_DOUBLE : -LIMBS_PER_DOUBLE);
      EXP(df) = __gmp_extract_double(darray, ABS(d));

      return gpgmp::internal::mpfArrayRoutines::gpmpf_cmp_array_idx_to_mpf_t(f, df);
    }

  }
}