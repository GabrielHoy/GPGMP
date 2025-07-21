#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_add(mpf_array_idx r, mpf_array_idx u, mpf_array_idx v)
    {
      MPF_ARRAY_ASSERT_OP_AVAILABLE(r.array, OP_ADD);
      mp_srcptr up, vp;
      mp_ptr rp;
      mp_size_t usize, vsize, rsize;
      mp_size_t prec;
      mp_exp_t uexp;
      mp_size_t ediff;
      mp_limb_t cy;
      int negate;
      mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];
      vsize = MPF_ARRAY_SIZES(v.array)[v.idx];

      /* Handle special cases that don't work in generic code below.  */
      if (usize == 0)
      {
      set_r_v_maybe:
        if ((r.array != v.array) || (r.idx != v.idx))
          gpmpf_set(r, v);
        return;
      }
      if (vsize == 0)
      {
        v = u;
        goto set_r_v_maybe;
      }

      /* If signs of U and V are different, perform subtraction.  */
      if ((usize ^ vsize) < 0)
      {
        __mpf_struct v_negated;
        v_negated._mp_size = -vsize;
        v_negated._mp_exp = MPF_ARRAY_EXPONENTS(v.array)[v.idx];
        v_negated._mp_d = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
        gpgmp::internal::mpfArrayRoutines::gpmpf_sub_mpf_t_from_array_idx(r, u, &v_negated);
        return;
      }

      /* Signs are now known to be the same.  */
      negate = usize < 0;

      /* Make U be the operand with the largest exponent.  */
      if (MPF_ARRAY_EXPONENTS(u.array)[u.idx] < MPF_ARRAY_EXPONENTS(v.array)[v.idx])
      {
        mpf_array_idx t;
        t = u;
        u = v;
        v = t;
        usize = MPF_ARRAY_SIZES(u.array)[u.idx];
        vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
      }

      usize = ABS(usize);
      vsize = ABS(vsize);
      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
      vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
      rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
      prec = r.array->userSpecifiedPrecisionLimbCount;
      uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
      ediff = uexp - MPF_ARRAY_EXPONENTS(v.array)[v.idx];

      /* If U extends beyond PREC, ignore the part that does.  */
      if (usize > prec)
      {
        up += usize - prec;
        usize = prec;
      }

      /* If V extends beyond PREC, ignore the part that does.
         Note that this may make vsize negative.  */
      if (vsize + ediff > prec)
      {
        vp += vsize + ediff - prec;
        vsize = prec - ediff;
      }

      /* Allocate temp space for the result.  Allocate
         just vsize + ediff later???  */

      if (ediff >= prec)
      {
        /* V completely cancelled.  */
        if (rp != up)
          MPN_COPY_INCR(rp, up, usize);
        rsize = usize;
      }
      else
      {
        /* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
        /* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

        if (usize > ediff)
        {
          /* U and V partially overlaps.  */
          if (vsize + ediff <= usize)
          {
            /* uuuu     */
            /*   v      */
            mp_size_t size;
            size = usize - ediff - vsize;
            MPN_COPY(scratchSpace, up, size);
            cy = gpgmp::mpnRoutines::gpmpn_add(scratchSpace + size, up + size, usize - size, vp, vsize);
            rsize = usize;
          }
          else
          {
            /* uuuu     */
            /*   vvvvv  */
            mp_size_t size;
            size = vsize + ediff - usize;
            MPN_COPY(scratchSpace, vp, size);
            cy = gpgmp::mpnRoutines::gpmpn_add(scratchSpace + size, up, usize, vp + size, usize - ediff);
            rsize = vsize + ediff;
          }
        }
        else
        {
          /* uuuu     */
          /*      vv  */
          mp_size_t size;
          size = vsize + ediff - usize;
          MPN_COPY(scratchSpace, vp, vsize);
          MPN_ZERO(scratchSpace + vsize, ediff - usize);
          MPN_COPY(scratchSpace + size, up, usize);
          cy = 0;
          rsize = size + usize;
        }

        MPN_COPY(rp, scratchSpace, rsize);
        rp[rsize] = cy;
        rsize += cy;
        uexp += cy;
      }

      MPF_ARRAY_SIZES(r.array)[r.idx] = negate ? -rsize : rsize;
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = uexp;
    }

  }

  namespace internal {
    namespace mpfArrayRoutines
    {
      ANYCALLER void gpmpf_add_mpf_t_to_array_idx(mpf_array_idx r, mpf_array_idx u, mpf_srcptr v)
      {
        mp_srcptr up, vp;
        mp_ptr rp;
        mp_size_t usize, vsize, rsize;
        mp_size_t prec;
        mp_exp_t uexp;
        mp_size_t ediff;
        mp_limb_t cy;
        int negate;
        mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

        usize = MPF_ARRAY_SIZES(u.array)[u.idx];
        vsize = v->_mp_size;

        /* Handle special cases that don't work in generic code below.  */
        if (usize == 0)
        {
          gpgmp::internal::mpfArrayRoutines::gpmpf_set_array_idx_to_mpf_t(r, v);
          return;
        }
        if (vsize == 0)
        {
          if ((r.array != u.array) || (r.idx != u.idx)) {
            gpgmp::mpfArrayRoutines::gpmpf_set(r, u);
          }
          return;
        }

        /* If signs of U and V are different, perform subtraction.  */
        if ((usize ^ vsize) < 0)
        {
          __mpf_struct v_negated;
          v_negated._mp_size = -vsize;
          v_negated._mp_exp = v->_mp_exp;
          v_negated._mp_d = v->_mp_d;
          gpgmp::internal::mpfArrayRoutines::gpmpf_sub_mpf_t_from_array_idx(r, u, &v_negated);
          return;
        }


        /* Signs are now known to be the same.  */
        negate = usize < 0;

        mpf_srcptr effectiveU;
        mpf_t effectiveV;

        /* Make U be the operand with the largest exponent.  */
        if (MPF_ARRAY_EXPONENTS(u.array)[u.idx] < v->_mp_exp)
        {
          effectiveU = v;
          effectiveV->_mp_exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
          effectiveV->_mp_size = MPF_ARRAY_SIZES(u.array)[u.idx];
          effectiveV->_mp_prec = u.array->userSpecifiedPrecisionLimbCount;
          effectiveV->_mp_d = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

          usize = effectiveU->_mp_size;
          vsize = effectiveV->_mp_size;
        }

        usize = ABS(usize);
        vsize = ABS(vsize);
        up = effectiveU ? PTR(effectiveU) : MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
        vp = PTR(effectiveV ? effectiveV : v);
        rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
        prec = r.array->userSpecifiedPrecisionLimbCount;
        uexp = effectiveU ? effectiveU->_mp_exp : MPF_ARRAY_EXPONENTS(u.array)[u.idx];
        ediff = (effectiveU ? effectiveU->_mp_exp : MPF_ARRAY_EXPONENTS(u.array)[u.idx]) - EXP(effectiveV ? effectiveV : v);

        /* If U extends beyond PREC, ignore the part that does.  */
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }

        /* If V extends beyond PREC, ignore the part that does.
           Note that this may make vsize negative.  */
        if (vsize + ediff > prec)
        {
          vp += vsize + ediff - prec;
          vsize = prec - ediff;
        }

        /* Allocate temp space for the result.  Allocate
           just vsize + ediff later???  */

        if (ediff >= prec)
        {
          /* V completely cancelled.  */
          if (rp != up)
            MPN_COPY_INCR(rp, up, usize);
          rsize = usize;
        }
        else
        {
          /* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
          /* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

          if (usize > ediff)
          {
            /* U and V partially overlaps.  */
            if (vsize + ediff <= usize)
            {
              /* uuuu     */
              /*   v      */
              mp_size_t size;
              size = usize - ediff - vsize;
              MPN_COPY(scratchSpace, up, size);
              cy = gpgmp::mpnRoutines::gpmpn_add(scratchSpace + size, up + size, usize - size, vp, vsize);
              rsize = usize;
            }
            else
            {
              /* uuuu     */
              /*   vvvvv  */
              mp_size_t size;
              size = vsize + ediff - usize;
              MPN_COPY(scratchSpace, vp, size);
              cy = gpgmp::mpnRoutines::gpmpn_add(scratchSpace + size, up, usize, vp + size, usize - ediff);
              rsize = vsize + ediff;
            }
          }
          else
          {
            /* uuuu     */
            /*      vv  */
            mp_size_t size;
            size = vsize + ediff - usize;
            MPN_COPY(scratchSpace, vp, vsize);
            MPN_ZERO(scratchSpace + vsize, ediff - usize);
            MPN_COPY(scratchSpace + size, up, usize);
            cy = 0;
            rsize = size + usize;
          }

          MPN_COPY(rp, scratchSpace, rsize);
          rp[rsize] = cy;
          rsize += cy;
          uexp += cy;
        }

        MPF_ARRAY_SIZES(r.array)[r.idx] = negate ? -rsize : rsize;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = uexp;
      }
    }

    namespace mpfRoutines
    {
      ANYCALLER void gpmpf_add_mpf_array_idx_to_mpf_array_idx(mpf_ptr r, mpf_array_idx u, mpf_array_idx v, mp_limb_t* scratchSpace)
      {
        mp_srcptr up, vp;
        mp_ptr rp;
        mp_size_t usize, vsize, rsize;
        mp_size_t prec;
        mp_exp_t uexp;
        mp_size_t ediff;
        mp_limb_t cy;
        int negate;

        usize = MPF_ARRAY_SIZES(u.array)[u.idx];
        vsize = MPF_ARRAY_SIZES(v.array)[v.idx];

        /* Handle special cases that don't work in generic code below.  */
        if (usize == 0)
        {
        set_r_v_maybe:
          gpgmp::internal::mpfRoutines::gpmpf_set_mpf_t_to_array_idx(r, v);
          return;
        }
        if (vsize == 0)
        {
          v = u;
          goto set_r_v_maybe;
        }

        /* If signs of U and V are different, perform subtraction.  */
        if ((usize ^ vsize) < 0)
        {
          gpgmp::mpfArrayRoutines::gpmpf_neg(v, v);
          //v is now negated, so we can use it directly in our subtraction.
          gpgmp::internal::mpfRoutines::gpmpf_sub_mpf_array_idx_from_mpf_array_idx(r, u, v, scratchSpace);
          //...now that the subtraction is done, we need to negate v again to ensure it remains unchanged overall.
          gpgmp::mpfArrayRoutines::gpmpf_neg(v, v);
          return;
        }

        /* Signs are now known to be the same.  */
        negate = usize < 0;

        /* Make U be the operand with the largest exponent.  */
        if (MPF_ARRAY_EXPONENTS(u.array)[u.idx] < MPF_ARRAY_EXPONENTS(v.array)[v.idx])
        {
          mpf_array_idx t;
          t = u;
          u = v;
          v = t;
          usize = MPF_ARRAY_SIZES(u.array)[u.idx];
          vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
        }

        usize = ABS(usize);
        vsize = ABS(vsize);
        up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
        vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
        rp = PTR(r);
        prec = PREC(r);
        uexp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
        ediff = uexp - MPF_ARRAY_EXPONENTS(v.array)[v.idx];

        /* If U extends beyond PREC, ignore the part that does.  */
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }

        /* If V extends beyond PREC, ignore the part that does.
          Note that this may make vsize negative.  */
        if (vsize + ediff > prec)
        {
          vp += vsize + ediff - prec;
          vsize = prec - ediff;
        }

        /* Allocate temp space for the result.  Allocate
          just vsize + ediff later???  */

        if (ediff >= prec)
        {
          /* V completely cancelled.  */
          if (rp != up)
            MPN_COPY_INCR(rp, up, usize);
          rsize = usize;
        }
        else
        {
          /* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
          /* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

          if (usize > ediff)
          {
            /* U and V partially overlaps.  */
            if (vsize + ediff <= usize)
            {
              /* uuuu     */
              /*   v      */
              mp_size_t size;
              size = usize - ediff - vsize;
              MPN_COPY(scratchSpace, up, size);
              cy = gpgmp::mpnRoutines::gpmpn_add(scratchSpace + size, up + size, usize - size, vp, vsize);
              rsize = usize;
            }
            else
            {
              /* uuuu     */
              /*   vvvvv  */
              mp_size_t size;
              size = vsize + ediff - usize;
              MPN_COPY(scratchSpace, vp, size);
              cy = gpgmp::mpnRoutines::gpmpn_add(scratchSpace + size, up, usize, vp + size, usize - ediff);
              rsize = vsize + ediff;
            }
          }
          else
          {
            /* uuuu     */
            /*      vv  */
            mp_size_t size;
            size = vsize + ediff - usize;
            MPN_COPY(scratchSpace, vp, vsize);
            MPN_ZERO(scratchSpace + vsize, ediff - usize);
            MPN_COPY(scratchSpace + size, up, usize);
            cy = 0;
            rsize = size + usize;
          }

          MPN_COPY(rp, scratchSpace, rsize);
          rp[rsize] = cy;
          rsize += cy;
          uexp += cy;
        }

        SIZ(r) = negate ? -rsize : rsize;
        EXP(r) = uexp;
      }

    }
  }
}