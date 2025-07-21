#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER unsigned long
    gpmpf_get_ui(mpf_array_idx f) __GMP_NOTHROW
    {
      mp_size_t size;
      mp_exp_t exp;
      mp_srcptr fp;
      mp_limb_t fl;

      exp = MPF_ARRAY_EXPONENTS(f.array)[f.idx];
      size = MPF_ARRAY_SIZES(f.array)[f.idx];
      fp = MPF_ARRAY_DATA_AT_IDX(f.array, f.idx);

      fl = 0;
      if (exp > 0)
      {
        /* there are some limbs above the radix point */

        size = ABS(size);
        if (size >= exp)
          fl = fp[size - exp];

#if BITS_PER_ULONG > GMP_NUMB_BITS
        if (exp > 1 && size + 1 >= exp)
          fl += (fp[size - exp + 1] << GMP_NUMB_BITS);
#endif
      }

      return (unsigned long)fl;
    }

  }
}