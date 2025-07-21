#include <stdio.h> /* for NULL */
#include "GPGMP/gpgmp-impl.cuh"
#include "GPGMP/longlong.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_ui_div(mpf_array_idx r, unsigned long int u, mpf_array_idx v)
    {
      MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_UI_DIV);
      mp_srcptr vp;
      mp_ptr rp, tp, remp, new_vp;
      mp_size_t vsize;
      mp_size_t rsize, prospective_rsize, zeros, tsize, high_zero;
      mp_size_t sign_quotient;
      mp_size_t prec;
      mp_exp_t rexp;
      mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

      vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
      sign_quotient = vsize;

      if (UNLIKELY(vsize == 0))
        DIVIDE_BY_ZERO;

      if (UNLIKELY(u == 0))
      {
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      vsize = ABS(vsize);
      prec = r.array->userSpecifiedPrecisionLimbCount;

      rexp = 1 - MPF_ARRAY_EXPONENTS(v.array)[v.idx] + 1;

      rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
      vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);

      prospective_rsize = 1 - vsize + 1; /* quot from using given u,v sizes */
      rsize = prec + 1;                  /* desired quot size */

      zeros = rsize - prospective_rsize; /* padding u to give rsize */
      tsize = 1 + zeros;                 /* u with zeros */

      if (WANT_TMP_DEBUG)
      {
        /* separate alloc blocks, for malloc debugging */
        remp = scratchSpace;
        scratchSpace += vsize;
        tp = scratchSpace;
        scratchSpace += tsize;
        new_vp = NULL;
        if (rp == vp)
        {
          new_vp = scratchSpace;
          scratchSpace += vsize;
        }
      }
      else
      {
        /* one alloc with calculated size, for efficiency */
        remp = scratchSpace;
        scratchSpace += vsize;
        tp = scratchSpace;
        scratchSpace += tsize;
        new_vp = scratchSpace;
        if (rp == vp)
        {
          scratchSpace += vsize;
        }
      }

      /* ensure divisor doesn't overlap quotient */
      if (rp == vp)
      {
        MPN_COPY(new_vp, vp, vsize);
        vp = new_vp;
      }

      MPN_ZERO(tp, tsize - 1);

      tp[tsize - 1] = u & GMP_NUMB_MASK;
#if BITS_PER_ULONG > GMP_NUMB_BITS
      if (u > GMP_NUMB_MAX)
      {
        /* tsize-vsize+1 == rsize, so tsize >= rsize.  rsize == prec+1 >= 2,
           so tsize >= 2, hence there's room for 2-limb u with nails */
        ASSERT(tsize >= 2);
        tp[tsize - 1] = u >> GMP_NUMB_BITS;
        tp[tsize - 2] = u & GMP_NUMB_MASK;
        rexp++;
      }
#endif

      ASSERT(tsize - vsize + 1 == rsize);
      gpgmp::mpnRoutines::gpmpn_tdiv_qr(rp, remp, (mp_size_t)0, tp, tsize, vp, vsize, scratchSpace);

      /* strip possible zero high limb */
      high_zero = (rp[rsize - 1] == 0);
      rsize -= high_zero;
      rexp -= high_zero;

      MPF_ARRAY_SIZES(r.array)[r.idx] = sign_quotient >= 0 ? rsize : -rsize;
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = rexp;
    }

  }
}