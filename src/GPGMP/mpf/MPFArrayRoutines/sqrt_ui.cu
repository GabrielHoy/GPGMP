#include <stdio.h>
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

#define U2 (GMP_NUMB_BITS < BITS_PER_ULONG)

    ANYCALLER void gpmpf_sqrt_ui(mpf_array_idx r, unsigned long int u)
    {
      MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_SQRT_UI);
      mp_size_t rsize, zeros;
      mp_size_t prec;
      mp_limb_t *scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

      if (UNLIKELY(u <= 1))
      {
        MPF_ARRAY_SIZES(r.array)[r.idx] = MPF_ARRAY_EXPONENTS(r.array)[r.idx] = u;
        *MPF_ARRAY_DATA_AT_IDX(r.array, r.idx) = u;
        return;
      }

      prec = r.array->userSpecifiedPrecisionLimbCount;
      zeros = 2 * prec - 2;
      rsize = zeros + 1 + U2;

      MPN_ZERO(scratchSpace, zeros);
      scratchSpace[zeros] = u & GMP_NUMB_MASK;

#if U2
      {
        mp_limb_t uhigh = u >> GMP_NUMB_BITS;
        scratchSpace[zeros + 1] = uhigh;
        rsize -= (uhigh == 0);
      }
#endif

      mp_limb_t* afterScratchSpace = scratchSpace + zeros;
      gpgmp::mpnRoutines::gpmpn_sqrtrem(MPF_ARRAY_DATA_AT_IDX(r.array, r.idx), NULL, scratchSpace, rsize, afterScratchSpace);

      MPF_ARRAY_SIZES(r.array)[r.idx] = prec;
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 1;
    }

  }
}