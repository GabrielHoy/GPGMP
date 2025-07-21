#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER int
    gpmpf_cmp_ui(mpf_array_idx u, unsigned long int vval) __GMP_NOTHROW
    {
      mp_srcptr up;
      mp_size_t usize;
      mp_exp_t uexp;
      mp_limb_t ulimb;

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];

      /* 1. Is U negative?  */
      if (usize < 0)
        return -1;
      /* We rely on usize being non-negative in the code that follows.  */

      if (vval == 0)
        return usize != 0;

      /* 2. Are the exponents different (V's exponent == 1)?  */
      uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];

#if GMP_NAIL_BITS != 0
      if (uexp != 1 + (vval > GMP_NUMB_MAX))
        return (uexp < 1 + (vval > GMP_NUMB_MAX)) ? -1 : 1;
#else
      if (uexp != 1)
        return (uexp < 1) ? -1 : 1;
#endif

      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

      ASSERT(usize > 0);
      ulimb = up[--usize];
#if GMP_NAIL_BITS != 0
      if (uexp == 2)
      {
        if ((ulimb >> GMP_NAIL_BITS) != 0)
          return 1;
        ulimb = (ulimb << GMP_NUMB_BITS);
        if (usize != 0)
          ulimb |= up[--usize];
      }
#endif

      /* 3. Compare the most significant mantissa limb with V.  */
      if (ulimb != vval)
        return (ulimb < vval) ? -1 : 1;

      /* Ignore zeroes at the low end of U.  */
      for (; *up == 0; ++up)
        --usize;

      /* 4. Now, if the number of limbs are different, we have a difference
         since we have made sure the trailing limbs are not zero.  */
      return (usize > 0);
    }

  }
}