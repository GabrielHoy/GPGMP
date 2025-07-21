#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* Not done:

       No attempt is made to identify an overlap u==v.  The result will be
       correct (1.0), but a full actual division is done whereas of course
       x/x==1 needs no work.  Such a call is not a sensible thing to make, and
       it's left to an application to notice and optimize if it might arise
       somehow through pointer aliasing or whatever.

       Enhancements:

       The high quotient limb is non-zero when high{up,vsize} >= {vp,vsize}.  We
       could make that comparison and use qsize==prec instead of qsize==prec+1,
       to save one limb in the division.

       If r==u but the size is enough bigger than prec that there won't be an
       overlap between quotient and dividend in mpn_div_q, then we can avoid
       copying up,usize.  This would only arise from a prec reduced with
       mpf_set_prec_raw and will be pretty unusual, but might be worthwhile if
       it could be worked into the copy_u decision cleanly.  */



    ANYCALLER void gpmpf_div(mpf_array_idx r, mpf_array_idx u, mpf_array_idx v)
    {
      MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_DIV);
      mp_srcptr up, vp;
      mp_ptr rp, new_vp;
      mp_size_t usize, vsize, rsize, prospective_rsize, tsize, zeros;
      mp_size_t sign_quotient, prec, high_zero, chop;
      mp_exp_t rexp;
      int copy_u;
      mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];
      vsize = MPF_ARRAY_SIZES(v.array)[v.idx];

      if (UNLIKELY(vsize == 0))
        DIVIDE_BY_ZERO;

      if (usize == 0)
      {
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      sign_quotient = usize ^ vsize;
      usize = ABS(usize);
      vsize = ABS(vsize);
      prec = r.array->userSpecifiedPrecisionLimbCount;

      rexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx] - MPF_ARRAY_EXPONENTS(v.array)[v.idx] + 1;

      rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
      vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);

      prospective_rsize = usize - vsize + 1; /* quot from using given u,v sizes */
      rsize = prec + 1;                      /* desired quot */

      zeros = rsize - prospective_rsize; /* padding u to give rsize */
      copy_u = (zeros > 0 || rp == up);  /* copy u if overlap or padding */

      chop = MAX(-zeros, 0); /* negative zeros means shorten u */
      up += chop;
      usize -= chop;
      zeros += chop; /* now zeros >= 0 */

      tsize = usize + zeros; /* size for possible copy of u */

      int scratchSpaceOffsetForNewVP = 0;
      /* copy and possibly extend u if necessary */
      if (copy_u)
      {
        scratchSpaceOffsetForNewVP = tsize + 1; /* +1 for mpn_div_q's scratch needs */
        MPN_ZERO(scratchSpace, zeros);
        MPN_COPY(scratchSpace + zeros, up, usize);
        up = scratchSpace;
        usize = tsize;
      }
      else
      {
        scratchSpaceOffsetForNewVP = usize + 1;
      }

      /* ensure divisor doesn't overlap quotient */
      if (rp == vp)
      {
        new_vp = scratchSpace + scratchSpaceOffsetForNewVP;
        MPN_COPY(new_vp, vp, vsize);
        vp = new_vp;
      }

      ASSERT(usize - vsize + 1 == rsize);
      gpgmp::mpnRoutines::gpmpn_div_q(rp, up, usize, vp, vsize, scratchSpace, scratchSpace + scratchSpaceOffsetForNewVP + vsize);

      /* strip possible zero high limb */
      high_zero = (rp[rsize - 1] == 0);
      rsize -= high_zero;
      rexp -= high_zero;

      MPF_ARRAY_SIZES(r.array)[r.idx] = sign_quotient >= 0 ? rsize : -rsize;
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = rexp;
    }

  }

  namespace internal
  {
    namespace mpfArrayRoutines
    {
      ANYCALLER void gpmpf_div_mpf_t_by_mpf_array_idx(mpf_array_idx r, mpf_srcptr u, mpf_array_idx v, mp_limb_t* scratchSpace)
      {
        mp_srcptr up, vp;
        mp_ptr rp, new_vp;
        mp_size_t usize, vsize, rsize, prospective_rsize, tsize, zeros;
        mp_size_t sign_quotient, prec, high_zero, chop;
        mp_exp_t rexp;
        int copy_u;

        usize = SIZ(u);
        vsize = MPF_ARRAY_SIZES(v.array)[v.idx];

        if (UNLIKELY(vsize == 0))
          DIVIDE_BY_ZERO;

        if (usize == 0)
        {
          MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
          MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
          return;
        }

        sign_quotient = usize ^ vsize;
        usize = ABS(usize);
        vsize = ABS(vsize);
        prec = r.array->userSpecifiedPrecisionLimbCount;

        rexp = EXP(u) - MPF_ARRAY_EXPONENTS(v.array)[v.idx] + 1;

        rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
        up = PTR(u);
        vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);

        prospective_rsize = usize - vsize + 1; /* quot from using given u,v sizes */
        rsize = prec + 1;                      /* desired quot */

        zeros = rsize - prospective_rsize; /* padding u to give rsize */
        copy_u = (zeros > 0 || rp == up);  /* copy u if overlap or padding */

        chop = MAX(-zeros, 0); /* negative zeros means shorten u */
        up += chop;
        usize -= chop;
        zeros += chop; /* now zeros >= 0 */

        tsize = usize + zeros; /* size for possible copy of u */

        int scratchSpaceOffsetForNewVP = 0;
        /* copy and possibly extend u if necessary */
        if (copy_u)
        {
          scratchSpaceOffsetForNewVP = tsize + 1; /* +1 for mpn_div_q's scratch needs */
          MPN_ZERO(scratchSpace, zeros);
          MPN_COPY(scratchSpace + zeros, up, usize);
          up = scratchSpace;
          usize = tsize;
        }
        else
        {
          scratchSpaceOffsetForNewVP = usize + 1;
        }

        /* ensure divisor doesn't overlap quotient */
        if (rp == vp)
        {
          new_vp = scratchSpace + scratchSpaceOffsetForNewVP;
          MPN_COPY(new_vp, vp, vsize);
          vp = new_vp;
        }

        ASSERT(usize - vsize + 1 == rsize);
        gpgmp::mpnRoutines::gpmpn_div_q(rp, up, usize, vp, vsize, scratchSpace, scratchSpace + scratchSpaceOffsetForNewVP + vsize);

        /* strip possible zero high limb */
        high_zero = (rp[rsize - 1] == 0);
        rsize -= high_zero;
        rexp -= high_zero;

        MPF_ARRAY_SIZES(r.array)[r.idx] = sign_quotient >= 0 ? rsize : -rsize;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = rexp;
      }
    }
  }
}