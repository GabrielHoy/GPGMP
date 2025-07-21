#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_random2(mpf_array_idx x, mp_size_t xs, mp_exp_t exp)
    {
      mp_size_t xn;
      mp_size_t prec;
      mp_limb_t elimb;

      xn = ABS(xs);
      prec = x.array->userSpecifiedPrecisionLimbCount;

      if (xn == 0)
      {
        MPF_ARRAY_EXPONENTS(x.array)[x.idx] = 0;
        MPF_ARRAY_SIZES(x.array)[x.idx] = 0;
        return;
      }

      if (xn > prec + 1)
        xn = prec + 1;

      /* General random mantissa.  */
      gpgmp::mpnRoutines::gpmpn_random2(MPF_ARRAY_DATA_AT_IDX(x.array, x.idx), xn);

      /* Generate random exponent.  */
      _gmp_rand(&elimb, RANDS, GMP_NUMB_BITS);
      exp = ABS(exp);
      exp = elimb % (2 * exp + 1) - exp;

      MPF_ARRAY_EXPONENTS(x.array)[x.idx] = exp;
      MPF_ARRAY_SIZES(x.array)[x.idx] = xs < 0 ? -xn : xn;
    }

  }
}