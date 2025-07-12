#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER int
    mpf_fits_sshort_p(mpf_array_idx f) __GMP_NOTHROW
    {
      mp_size_t fs, fn;
      mp_srcptr fp;
      mp_exp_t exp;
      mp_limb_t fl;

      exp = MPF_ARRAY_EXPONENTS(f.array)[f.idx];
      if (exp < 1)
        return 1;

      fs = MPF_ARRAY_SIZES(f.array)[f.idx];
      fp = MPF_ARRAY_DATA_AT_IDX(f.array, f.idx);
      fn = ABS(fs);

      if (exp == 1)
      {
        fl = fp[fn - 1];
      }
#if GMP_NAIL_BITS != 0
      else if (exp == 2 && SHRT_MAX > GMP_NUMB_MAX)
      {
        fl = fp[fn - 1];
        if ((fl >> GMP_NAIL_BITS) != 0)
          return 0;
        fl = (fl << GMP_NUMB_BITS);
        if (fn >= 2)
          fl |= fp[fn - 2];
      }
#endif
      else
        return 0;

      return fl <= (fs >= 0 ? static_cast<mp_limb_t>(SHRT_MAX) : NEG_CAST(mp_limb_t, SHRT_MIN));
    }

  }
}