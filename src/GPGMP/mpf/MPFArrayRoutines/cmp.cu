#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER int
    gpmpf_cmp(mpf_array_idx u, mpf_array_idx v) __GMP_NOTHROW
    {
      mp_srcptr up, vp;
      mp_size_t usize, vsize;
      mp_exp_t uexp, vexp;
      int cmp;
      int usign;

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];
      vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
      usign = usize >= 0 ? 1 : -1;

      /* 1. Are the signs different?  */
      if ((usize ^ vsize) >= 0)
      {
        /* U and V are both non-negative or both negative.  */
        if (usize == 0)
          /* vsize >= 0 */
          return -(vsize != 0);
        if (vsize == 0)
          /* usize >= 0 */
          return usize != 0;
        /* Fall out.  */
      }
      else
      {
        /* Either U or V is negative, but not both.  */
        return usign;
      }

      /* U and V have the same sign and are both non-zero.  */

      uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
      vexp = MPF_ARRAY_EXPONENTS(v.array)[v.idx];

      /* 2. Are the exponents different?  */
      if (uexp > vexp)
        return usign;
      if (uexp < vexp)
        return -usign;

      usize = ABS(usize);
      vsize = ABS(vsize);

      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
      vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);

#define STRICT_MPF_NORMALIZATION 0
#if !STRICT_MPF_NORMALIZATION
      /* Ignore zeroes at the low end of U and V.  */
      do
      {
        mp_limb_t tl;
        tl = up[0];
        MPN_STRIP_LOW_ZEROS_NOT_ZERO(up, usize, tl);
        tl = vp[0];
        MPN_STRIP_LOW_ZEROS_NOT_ZERO(vp, vsize, tl);
      } while (0);
#endif

      if (usize > vsize)
      {
        cmp = gpgmp::mpnRoutines::gpmpn_cmp(up + usize - vsize, vp, vsize);
        /* if (cmp == 0) */
        /*	return usign; */
        ++cmp;
      }
      else if (vsize > usize)
      {
        cmp = gpgmp::mpnRoutines::gpmpn_cmp(up, vp + vsize - usize, usize);
        /* if (cmp == 0) */
        /*	return -usign; */
      }
      else
      {
        cmp = gpgmp::mpnRoutines::gpmpn_cmp(up, vp, usize);
        if (cmp == 0)
          return 0;
      }
      return cmp > 0 ? usign : -usign;
    }

  }

  namespace internal
  {
    namespace mpfArrayRoutines
    {
      ANYCALLER int gpmpf_cmp_array_idx_to_mpf_t(mpf_array_idx u, mpf_srcptr v) __GMP_NOTHROW
      {
        mp_srcptr up, vp;
        mp_size_t usize, vsize;
        mp_exp_t uexp, vexp;
        int cmp;
        int usign;

        usize = MPF_ARRAY_SIZES(u.array)[u.idx];
        vsize = SIZ(v);
        usign = usize >= 0 ? 1 : -1;

        /* 1. Are the signs different?  */
        if ((usize ^ vsize) >= 0)
        {
          /* U and V are both non-negative or both negative.  */
          if (usize == 0)
            /* vsize >= 0 */
            return -(vsize != 0);
          if (vsize == 0)
            /* usize >= 0 */
            return usize != 0;
          /* Fall out.  */
        }
        else
        {
          /* Either U or V is negative, but not both.  */
          return usign;
        }

        /* U and V have the same sign and are both non-zero.  */

        uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
        vexp = EXP(v);

        /* 2. Are the exponents different?  */
        if (uexp > vexp)
          return usign;
        if (uexp < vexp)
          return -usign;

        usize = ABS(usize);
        vsize = ABS(vsize);

        up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
        vp = PTR(v);

  #define STRICT_MPF_NORMALIZATION 0
  #if !STRICT_MPF_NORMALIZATION
        /* Ignore zeroes at the low end of U and V.  */
        do
        {
          mp_limb_t tl;
          tl = up[0];
          MPN_STRIP_LOW_ZEROS_NOT_ZERO(up, usize, tl);
          tl = vp[0];
          MPN_STRIP_LOW_ZEROS_NOT_ZERO(vp, vsize, tl);
        } while (0);
  #endif

        if (usize > vsize)
        {
          cmp = gpgmp::mpnRoutines::gpmpn_cmp(up + usize - vsize, vp, vsize);
          /* if (cmp == 0) */
          /*	return usign; */
          ++cmp;
        }
        else if (vsize > usize)
        {
          cmp = gpgmp::mpnRoutines::gpmpn_cmp(up, vp + vsize - usize, usize);
          /* if (cmp == 0) */
          /*	return -usign; */
        }
        else
        {
          cmp = gpgmp::mpnRoutines::gpmpn_cmp(up, vp, usize);
          if (cmp == 0)
            return 0;
        }
        return cmp > 0 ? usign : -usign;
      }

    }

  }
}