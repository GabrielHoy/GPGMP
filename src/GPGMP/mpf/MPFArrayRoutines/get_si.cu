#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER long
    gpmpf_get_si(mpf_array_idx f) __GMP_NOTHROW
    {
      mp_exp_t exp;
      mp_size_t size, abs_size;
      mp_srcptr fp;
      mp_limb_t fl;

      exp = MPF_ARRAY_EXPONENTS(f.array)[f.idx];
      size = MPF_ARRAY_SIZES(f.array)[f.idx];
      fp = MPF_ARRAY_DATA_AT_IDX(f.array, f.idx);

      if (exp <= 0)
        return 0L;


      fl = 0;
      abs_size = ABS(size);
      if (abs_size >= exp)
        fl = fp[abs_size - exp];

#if BITS_PER_ULONG > GMP_NUMB_BITS
      if (exp > 1 && abs_size + 1 >= exp)
        fl |= fp[abs_size - exp + 1] << GMP_NUMB_BITS;
#endif

      if (size > 0)
        return fl & LONG_MAX;
      else
        return -1 - (long)((fl - 1) & LONG_MAX);
    }

  }
}