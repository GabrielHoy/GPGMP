#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_swap(mpf_array_idx toSwap, mpf_array_idx swapWith) __GMP_NOTHROW
    {
      mp_size_t tprec;
      mp_size_t tsiz;
      mp_exp_t texp;

      tprec = toSwap.array->userSpecifiedPrecisionLimbCount;
      toSwap.array->userSpecifiedPrecisionLimbCount = swapWith.array->userSpecifiedPrecisionLimbCount;
      swapWith.array->userSpecifiedPrecisionLimbCount = tprec;

      tsiz = MPF_ARRAY_SIZES(toSwap.array)[toSwap.idx];
      MPF_ARRAY_SIZES(toSwap.array)[toSwap.idx] = MPF_ARRAY_SIZES(swapWith.array)[swapWith.idx];
      MPF_ARRAY_SIZES(swapWith.array)[swapWith.idx] = tsiz;

      texp = MPF_ARRAY_EXPONENTS(toSwap.array)[toSwap.idx];
      MPF_ARRAY_EXPONENTS(toSwap.array)[toSwap.idx] = MPF_ARRAY_EXPONENTS(swapWith.array)[swapWith.idx];
      MPF_ARRAY_EXPONENTS(swapWith.array)[swapWith.idx] = texp;

      // In normal GMP, the limb data is simply a pointer to the data array.
      // GMP can swap these pointers directly w/o needing to copy data, but GPGMP's arrays don't work that way.
      // Unfortunately, this means that we need to actually copy all the data.
      mp_limb_t* startOfDataToSwap = MPF_ARRAY_DATA_AT_IDX(toSwap.array, toSwap.idx);
      mp_limb_t* startOfDataToSwapWith = MPF_ARRAY_DATA_AT_IDX(swapWith.array, swapWith.idx);
      for (int limbIdx = 0; limbIdx < tsiz; limbIdx++) {
        mp_limb_t temp = startOfDataToSwap[limbIdx];
        startOfDataToSwap[limbIdx] = startOfDataToSwapWith[limbIdx];
        startOfDataToSwapWith[limbIdx] = temp;
      }
    }

  }
}