#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {


    ANYCALLER int
    gpmpf_eq(mpf_array_idx u, mpf_array_idx v, mp_bitcnt_t n_bits)
    {
      mp_srcptr up, vp, p;
      mp_size_t usize, vsize, minsize, maxsize, n_limbs, i, size;
      mp_exp_t uexp, vexp;
      mp_limb_t diff;
      int cnt;

      uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
      vexp = MPF_ARRAY_EXPONENTS(v.array)[v.idx];

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];
      vsize = MPF_ARRAY_SIZES(v.array)[v.idx];

      /* 1. Are the signs different?  */
      if ((usize ^ vsize) >= 0)
      {
        /* U and V are both non-negative or both negative.  */
        if (usize == 0)
          return vsize == 0;
        if (vsize == 0)
          return 0;

        /* Fall out.  */
      }
      else
      {
        /* Either U or V is negative, but not both.  */
        return 0;
      }

      /* U and V have the same sign and are both non-zero.  */

      /* 2. Are the exponents different?  */
      if (uexp != vexp)
        return 0;

      usize = ABS(usize);
      vsize = ABS(vsize);

      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
      vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);

      up += usize; /* point just above most significant limb */
      vp += vsize; /* point just above most significant limb */

      count_leading_zeros(cnt, up[-1]);
      if ((vp[-1] >> (GMP_LIMB_BITS - 1 - cnt)) != 1)
        return 0; /* msb positions different */

      n_bits += cnt - GMP_NAIL_BITS;
      n_limbs = (n_bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;

      usize = MIN(usize, n_limbs);
      vsize = MIN(vsize, n_limbs);

      minsize = MIN(usize, vsize);
      maxsize = usize + vsize - minsize;

      up -= minsize; /* point at most significant common limb */
      vp -= minsize; /* point at most significant common limb */

      /* Compare the most significant part which has explicit limbs for U and V. */
      for (i = minsize - 1; i > 0; i--)
      {
        if (up[i] != vp[i])
          return 0;
      }

      n_bits -= (maxsize - 1) * GMP_NUMB_BITS;

      size = maxsize - minsize;
      if (size != 0)
      {
        if (up[0] != vp[0])
          return 0;

        /* Now either U or V has its limbs consumed, i.e, continues with an
     infinite number of implicit zero limbs.  Check that the other operand
     has just zeros in the corresponding, relevant part.  */

        if (usize > vsize)
          p = up - size;
        else
          p = vp - size;

        for (i = size - 1; i > 0; i--)
        {
          if (p[i] != 0)
            return 0;
        }

        diff = p[0];
      }
      else
      {
        /* Both U or V has its limbs consumed.  */

        diff = up[0] ^ vp[0];
      }

      if (n_bits < GMP_NUMB_BITS)
        diff >>= GMP_NUMB_BITS - n_bits;

      return diff == 0;
    }

  }
}