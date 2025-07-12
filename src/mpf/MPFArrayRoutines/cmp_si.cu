#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER int
    gpmpf_cmp_si(mpf_array_idx u, long int vval) __GMP_NOTHROW
    {
      mp_srcptr up;
      mp_size_t usize;
      mp_exp_t uexp;
      mp_limb_t ulimb;
      int usign;
      unsigned long abs_vval;

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];

      /* 1. Are the signs different?  */
      if ((usize < 0) == (vval < 0)) /* don't use xor, type size may differ */
      {
        /* U and V are both non-negative or both negative.  */
        if (usize == 0)
          /* vval >= 0 */
          return -(vval != 0);
        if (vval == 0)
          /* usize >= 0 */
          return usize != 0;
        /* Fall out.  */
      }
      else
      {
        /* Either U or V is negative, but not both.  */
        return usize >= 0 ? 1 : -1;
      }

      /* U and V have the same sign and are both non-zero.  */

      /* 2. Are the exponents different (V's exponent == 1)?  */
      uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
      usign = usize >= 0 ? 1 : -1;
      usize = ABS(usize);
      abs_vval = ABS_CAST(unsigned long, vval);

#if GMP_NAIL_BITS != 0
      if (uexp != 1 + (abs_vval > GMP_NUMB_MAX))
        return (uexp < 1 + (abs_vval > GMP_NUMB_MAX)) ? -usign : usign;
#else
      if (uexp != 1)
        return (uexp < 1) ? -usign : usign;
#endif

      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

      ASSERT(usize > 0);
      ulimb = up[--usize];
#if GMP_NAIL_BITS != 0
      if (uexp == 2)
      {
        if ((ulimb >> GMP_NAIL_BITS) != 0)
          return usign;
        ulimb = (ulimb << GMP_NUMB_BITS);
        if (usize != 0)
          ulimb |= up[--usize];
      }
#endif

      /* 3. Compare the most significant mantissa limb with V.  */
      if (ulimb != abs_vval)
        return (ulimb < abs_vval) ? -usign : usign;

      /* Ignore zeroes at the low end of U.  */
      for (; *up == 0; ++up)
        --usize;

      /* 4. Now, if the number of limbs are different, we have a difference
         since we have made sure the trailing limbs are not zero.  */
      if (usize > 0)
        return usign;

      /* Wow, we got zero even if we tried hard to avoid it.  */
      return 0;
    }

  }
}