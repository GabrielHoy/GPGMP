#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_mul(mpf_array_idx r, mpf_array_idx u, mpf_array_idx v)
    {
      mp_size_t sign_product;
      mp_size_t prec = r.array->userSpecifiedPrecisionLimbCount;
      mp_size_t rsize;
      mp_limb_t cy_limb;
      mp_ptr rp;
      mp_size_t adj;
      mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);

      if ((u.array == v.array) && (u.idx == v.idx))
      {
        mp_srcptr up;
        mp_size_t usize;

        sign_product = 0;

        usize = ABS(MPF_ARRAY_SIZES(u.array)[u.idx]);

        up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }

        if (usize == 0)
        {
          MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
          MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0; /* ??? */
          return;
        }
        else
        {
          rsize = 2 * usize;

          gpgmp::mpnRoutines::gpmpn_sqr(scratchSpace, up, usize);
          cy_limb = scratchSpace[rsize - 1];
        }
      }
      else
      {
        mp_srcptr up, vp;
        mp_size_t usize, vsize;

        usize = MPF_ARRAY_SIZES(u.array)[u.idx];
        vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
        sign_product = usize ^ vsize;

        usize = ABS(usize);
        vsize = ABS(vsize);

        up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
        vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }
        if (vsize > prec)
        {
          vp += vsize - prec;
          vsize = prec;
        }

        if (usize == 0 || vsize == 0)
        {
          MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
          MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
          return;
        }
        else
        {
          rsize = usize + vsize;
          cy_limb = (usize >= vsize
                         ? gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, up, usize, vp, vsize)
                         : gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, vp, vsize, up, usize));
        }
      }

      adj = cy_limb == 0;
      rsize -= adj;
      prec++;
      if (rsize > prec)
      {
        scratchSpace += rsize - prec;
        rsize = prec;
      }
      rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
      MPN_COPY(rp, scratchSpace, rsize);
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = MPF_ARRAY_EXPONENTS(u.array)[u.idx] + MPF_ARRAY_EXPONENTS(v.array)[v.idx] - adj;
      MPF_ARRAY_SIZES(r.array)[r.idx] = sign_product >= 0 ? rsize : -rsize;
    }

  }

  namespace internal
  {
    namespace mpfArrayRoutines
    {

      ANYCALLER void gpmpf_mul_mpf_t_by_mpf_array_idx(mpf_array_idx r, mpf_srcptr u, mpf_array_idx v, mp_limb_t* scratchSpace)
      {
        mp_size_t sign_product;
        mp_size_t prec = r.array->userSpecifiedPrecisionLimbCount;
        mp_size_t rsize;
        mp_limb_t cy_limb;
        mp_ptr rp;
        mp_size_t adj;


        mp_srcptr up, vp;
        mp_size_t usize, vsize;

        usize = SIZ(u);
        vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
        sign_product = usize ^ vsize;

        usize = ABS(usize);
        vsize = ABS(vsize);

        up = PTR(u);
        vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }
        if (vsize > prec)
        {
          vp += vsize - prec;
          vsize = prec;
        }

        if (usize == 0 || vsize == 0)
        {
          MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
          MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
          return;
        }
        else
        {
          rsize = usize + vsize;
          cy_limb = (usize >= vsize
                        ? gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, up, usize, vp, vsize)
                        : gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, vp, vsize, up, usize));
        }


        adj = cy_limb == 0;
        rsize -= adj;
        prec++;
        if (rsize > prec)
        {
          scratchSpace += rsize - prec;
          rsize = prec;
        }
        rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
        MPN_COPY(rp, scratchSpace, rsize);
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = EXP(u) + MPF_ARRAY_EXPONENTS(v.array)[v.idx] - adj;
        MPF_ARRAY_SIZES(r.array)[r.idx] = sign_product >= 0 ? rsize : -rsize;
      }

      ANYCALLER void gpmpf_mul_mpf_t_by_mpf_t(mpf_array_idx r, mpf_srcptr u, mpf_srcptr v, mp_limb_t* scratchSpace)
      {
        mp_size_t sign_product;
        mp_size_t prec = r.array->userSpecifiedPrecisionLimbCount;
        mp_size_t rsize;
        mp_limb_t cy_limb;
        mp_ptr rp;
        mp_size_t adj;

        if (u == v)
        {
          mp_srcptr up;
          mp_size_t usize;

          sign_product = 0;

          usize = ABSIZ(u);

          up = PTR(u);
          if (usize > prec)
          {
            up += usize - prec;
            usize = prec;
          }

          if (usize == 0)
          {
            MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
            MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0; /* ??? */
            return;
          }
          else
          {
            rsize = 2 * usize;

            gpgmp::mpnRoutines::gpmpn_sqr(scratchSpace, up, usize);
            cy_limb = scratchSpace[rsize - 1];
          }
        }
        else
        {
          mp_srcptr up, vp;
          mp_size_t usize, vsize;

          usize = SIZ(u);
          vsize = SIZ(v);
          sign_product = usize ^ vsize;

          usize = ABS(usize);
          vsize = ABS(vsize);

          up = PTR(u);
          vp = PTR(v);
          if (usize > prec)
          {
            up += usize - prec;
            usize = prec;
          }
          if (vsize > prec)
          {
            vp += vsize - prec;
            vsize = prec;
          }

          if (usize == 0 || vsize == 0)
          {
            MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
            MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
            return;
          }
          else
          {
            rsize = usize + vsize;
            cy_limb = (usize >= vsize
                          ? gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, up, usize, vp, vsize)
                          : gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, vp, vsize, up, usize));
          }
        }

        adj = cy_limb == 0;
        rsize -= adj;
        prec++;
        if (rsize > prec)
        {
          scratchSpace += rsize - prec;
          rsize = prec;
        }
        rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
        MPN_COPY(rp, scratchSpace, rsize);
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = EXP(u) + EXP(v) - adj;
        MPF_ARRAY_SIZES(r.array)[r.idx] = sign_product >= 0 ? rsize : -rsize;
      }
    }

    namespace mpfRoutines
    {
      ANYCALLER void gpmpf_mul_mpf_t_by_mpf_array_idx(mpf_ptr r, mpf_srcptr u, mpf_array_idx v, mp_limb_t* scratchSpace)
      {
        mp_size_t sign_product;
        mp_size_t prec = PREC(r);
        mp_size_t rsize;
        mp_limb_t cy_limb;
        mp_ptr rp;
        mp_size_t adj;



        mp_srcptr up, vp;
        mp_size_t usize, vsize;

        usize = SIZ(u);
        vsize = MPF_ARRAY_SIZES(v.array)[v.idx];
        sign_product = usize ^ vsize;

        usize = ABS(usize);
        vsize = ABS(vsize);

        up = PTR(u);
        vp = MPF_ARRAY_DATA_AT_IDX(v.array, v.idx);
        if (usize > prec)
        {
          up += usize - prec;
          usize = prec;
        }
        if (vsize > prec)
        {
          vp += vsize - prec;
          vsize = prec;
        }

        if (usize == 0 || vsize == 0)
        {
          SIZ(r) = 0;
          EXP(r) = 0;
          return;
        }
        else
        {
          rsize = usize + vsize;
          cy_limb = (usize >= vsize
                          ? gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, up, usize, vp, vsize)
                          : gpgmp::mpnRoutines::gpmpn_mul(scratchSpace, vp, vsize, up, usize));
        }


        adj = cy_limb == 0;
        rsize -= adj;
        prec++;
        if (rsize > prec)
        {
          scratchSpace += rsize - prec;
          rsize = prec;
        }
        rp = PTR(r);
        MPN_COPY(rp, scratchSpace, rsize);
        EXP(r) = EXP(u) + MPF_ARRAY_EXPONENTS(v.array)[v.idx] - adj;
        SIZ(r) = sign_product >= 0 ? rsize : -rsize;
      }

    }
  }
}