#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_set_z(mpf_array_idx toSet, mpz_srcptr setFrom)
    {
      mp_ptr rp, up;
      mp_size_t size, asize;
      mp_size_t prec;

      prec = toSet.array->userSpecifiedPrecisionLimbCount + 1;
      size = SIZ(setFrom);
      asize = ABS(size);
      rp = MPF_ARRAY_DATA_AT_IDX(toSet.array, toSet.idx);
      up = PTR(setFrom);

      MPF_ARRAY_EXPONENTS(toSet.array)[toSet.idx] = asize;

      if (asize > prec)
      {
        up += asize - prec;
        asize = prec;
      }

      MPF_ARRAY_SIZES(toSet.array)[toSet.idx] = size >= 0 ? asize : -asize;
      MPN_COPY(rp, up, asize);
    }

  }
}