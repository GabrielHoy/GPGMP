#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* dir==1 for ceil, dir==-1 for floor

       Notice the use of prec+1 ensures mpf_ceil and mpf_floor are equivalent to
       mpf_set if u is already an integer.  */

    ANYCALLER static void __gpmpf_ceil_or_floor(REGPARM_2_1(mpf_array_idx, mpf_array_idx, int)) REGPARM_ATTR(1);
#define gpmpf_ceil_or_floor(r, u, dir) __gpmpf_ceil_or_floor(REGPARM_2_1(r, u, dir))

    REGPARM_ATTR(1)
    ANYCALLER static void gpmpf_ceil_or_floor(mpf_array_idx r, mpf_array_idx u, int dir)
    {
      mp_ptr rp, up, p;
      mp_size_t size, asize, prec;
      mp_exp_t exp;

      size = MPF_ARRAY_SIZES(u.array)[u.idx];
      if (size == 0)
      {
      zero:
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
      exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
      if (exp <= 0)
      {
        /* u is only a fraction */
        if ((size ^ dir) < 0)
          goto zero;
        rp[0] = 1;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 1;
        MPF_ARRAY_SIZES(r.array)[r.idx] = dir;
        return;
      }
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = exp;

      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
      asize = ABS(size);
      up += asize;

      /* skip fraction part of u */
      asize = MIN(asize, exp);

      /* don't lose precision in the copy */
      prec = r.array->userSpecifiedPrecisionLimbCount + 1;

      /* skip excess over target precision */
      asize = MIN(asize, prec);

      up -= asize;

      if ((size ^ dir) >= 0)
      {
        /* rounding direction matches sign, must increment if ignored part is
           non-zero */
        for (p = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx); p != up; p++)
        {
          if (*p != 0)
          {
            if (gpgmp::mpnRoutines::gpmpn_add_1(rp, up, asize, CNST_LIMB(1)))
            {
              /* was all 0xFF..FFs, which have become zeros, giving just
                 a carry */
              rp[0] = 1;
              asize = 1;
              MPF_ARRAY_EXPONENTS(r.array)[r.idx]++;
            }
            MPF_ARRAY_SIZES(r.array)[r.idx] = (size >= 0 ? asize : -asize);
            return;
          }
        }
      }

      MPF_ARRAY_SIZES(r.array)[r.idx] = (size >= 0 ? asize : -asize);
      if (rp != up)
        MPN_COPY_INCR(rp, up, asize);
    }

    ANYCALLER void
    gpmpf_ceil(mpf_array_idx r, mpf_array_idx u)
    {
      gpmpf_ceil_or_floor(r, u, 1);
    }

    ANYCALLER void
    gpmpf_floor(mpf_array_idx r, mpf_array_idx u)
    {
      gpmpf_ceil_or_floor(r, u, -1);
    }

  }
}