

#include <stdio.h> /* for NULL */
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void
    gpgpmpf_sqrt(mpf_array_idx r, mpf_array_idx u)
    {
      MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_SQRT);
      mp_size_t usize;
      mp_ptr up;
      mp_size_t prec, tsize;
      mp_exp_t uexp, expodd;
      mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];
      if (UNLIKELY(usize <= 0))
      {
        if (usize < 0)
          SQRT_OF_NEGATIVE;
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
      prec = MPF_ARRAY_SIZES(r.array)[r.idx];
      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

      expodd = (uexp & 1);
      tsize = 2 * prec - expodd;
      MPF_ARRAY_SIZES(r.array)[r.idx] = prec;
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = (uexp + expodd) / 2; /* ceil(uexp/2) */

      /* root size is ceil(tsize/2), this will be our desired "prec" limbs */
      ASSERT((tsize + 1) / 2 == prec);

      if (usize > tsize)
      {
        up += usize - tsize;
        usize = tsize;
        MPN_COPY(scratchSpace, up, tsize);
      }
      else
      {
        MPN_ZERO(scratchSpace, tsize - usize);
        MPN_COPY(scratchSpace + (tsize - usize), up, usize);
      }

      gpgmp::mpnRoutines::gpmpn_sqrtrem(MPF_ARRAY_DATA_AT_IDX(r.array, r.idx), NULL, scratchSpace, tsize, scratchSpace + (tsize - usize) + usize);
    }

  }
}