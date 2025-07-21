#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_set(mpf_array_idx set, mpf_array_idx copy)
    {
      mp_ptr rp, up;
      mp_size_t size, asize;
      mp_size_t prec;

      prec = set.array->userSpecifiedPrecisionLimbCount+1; /* lie not to lose precision in assignment */
      size = MPF_ARRAY_SIZES(copy.array)[copy.idx];
      asize = ABS(size);
      rp = MPF_ARRAY_DATA_AT_IDX(set.array, set.idx);
      up = MPF_ARRAY_DATA_AT_IDX(copy.array, copy.idx);

      if (asize > prec)
      {
        up += asize - prec;
        asize = prec;
      }

      MPF_ARRAY_EXPONENTS(set.array)[set.idx] = MPF_ARRAY_EXPONENTS(copy.array)[copy.idx];
      MPF_ARRAY_SIZES(set.array)[set.idx] = size >= 0 ? asize : -asize;
      MPN_COPY_INCR(rp, up, asize);
    }

  }

  namespace internal {

    namespace mpfArrayRoutines
    {
      ANYCALLER void
      gpmpf_set_array_idx_to_mpf_t(mpf_array_idx r, mpf_srcptr u)
      {
        mp_ptr rp, up;
        mp_size_t size, asize;
        mp_size_t prec;

        prec = r.array->userSpecifiedPrecisionLimbCount + 1; /* lie not to lose precision in assignment */
        size = u->_mp_size;
        asize = ABS(size);
        rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
        up = u->_mp_d;

        if (asize > prec)
        {
          up += asize - prec;
          asize = prec;
        }

        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = u->_mp_exp;
        MPF_ARRAY_SIZES(r.array)[r.idx] = size >= 0 ? asize : -asize;
        MPN_COPY_INCR(rp, up, asize);
      }
    }

    namespace mpfRoutines
    {
      ANYCALLER void
      gpmpf_set_mpf_t_to_array_idx(mpf_ptr r, mpf_array_idx u)
      {
        mp_ptr rp, up;
        mp_size_t size, asize;
        mp_size_t prec;

        prec = r->_mp_prec + 1; /* lie not to lose precision in assignment */
        size = MPF_ARRAY_SIZES(u.array)[u.idx];
        asize = ABS(size);
        rp = r->_mp_d;
        up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

        if (asize > prec)
        {
          up += asize - prec;
          asize = prec;
        }

        r->_mp_exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
        r->_mp_size = size >= 0 ? asize : -asize;
        MPN_COPY_INCR(rp, up, asize);
      }
    }

  }
}