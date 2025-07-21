#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_div_ui(mpf_array_idx r, mpf_array_idx u, unsigned long int v)
    {
      MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_DIV_UI);
      mp_srcptr up;
      mp_ptr rp, rtp;
      mp_size_t usize;
      mp_size_t rsize, tsize;
      mp_size_t sign_quotient;
      mp_size_t prec;
      mp_limb_t q_limb;
      mp_exp_t rexp;
      mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

      //TODO: Handle this case
#if BITS_PER_ULONG > GMP_NUMB_BITS /* avoid warnings about shift amount */
      #error "BITS_PER_ULONG > GMP_NUMB_BITS, this case is not supported in GPGMP yet..."
      if (v > GMP_NUMB_MAX)
      {
        mpf_t vf;
        mp_limb_t vl[2];
        SIZ(vf) = 2;
        EXP(vf) = 2;
        PTR(vf) = vl;
        vl[0] = v & GMP_NUMB_MASK;
        vl[1] = v >> GMP_NUMB_BITS;
        gpmpf_div(r, u, vf);
        return;
      }
#endif

      if (UNLIKELY(v == 0))
        DIVIDE_BY_ZERO;

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];

      if (usize == 0)
      {
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      sign_quotient = usize;
      usize = ABS(usize);
      prec = r.array->userSpecifiedPrecisionLimbCount;

      rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

      tsize = 1 + prec;

      if (usize > tsize)
      {
        up += usize - tsize;
        usize = tsize;
        rtp = scratchSpace;
      }
      else
      {
        MPN_ZERO(scratchSpace, tsize - usize);
        rtp = scratchSpace + (tsize - usize);
      }

      /* Move the dividend to the remainder.  */
      MPN_COPY(rtp, up, usize);

      gpgmp::mpnRoutines::gpmpn_divmod_1(rp, scratchSpace, tsize, (mp_limb_t)v);
      q_limb = rp[tsize - 1];

      rsize = tsize - (q_limb == 0);
      rexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx] - (q_limb == 0);
      MPF_ARRAY_SIZES(r.array)[r.idx] = sign_quotient >= 0 ? rsize : -rsize;
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = rexp;
    }

  }
}