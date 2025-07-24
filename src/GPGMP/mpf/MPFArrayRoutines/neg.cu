#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void
    gpmpf_neg(mpf_array_idx r, mpf_array_idx u)
    {
      mp_size_t size;

      size = -MPF_ARRAY_SIZES(u.array)[u.idx];
      if ((r.array != u.array) || (r.idx != u.idx))
      {
        mp_size_t prec;
        mp_size_t asize;
        mp_ptr rp, up;

        prec = r.array->userSpecifiedPrecisionLimbCount + 1;
        asize = ABS(size);
        rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
        up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

        if (asize > prec)
        {
          up += asize - prec;
          asize = prec;
        }

        MPN_COPY(rp, up, asize);
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
        size = size >= 0 ? asize : -asize;
      }
      MPF_ARRAY_SIZES(r.array)[r.idx] = size;
    }

  }

  namespace internal
  {
    namespace mpfArrayRoutines
    {
      ANYCALLER void gpmpf_neg_set_array_idx_from_mpf_t(mpf_array_idx r, mpf_srcptr u)
      {
        mp_size_t size;

        size = -u->_mp_size;



        mp_size_t prec;
        mp_size_t asize;
        mp_ptr rp, up;

        prec = r.array->userSpecifiedPrecisionLimbCount + 1; /* lie not to lose precision in assignment */
        asize = ABS(size);
        rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
        up = u->_mp_d;

        if (asize > prec)
        {
          up += asize - prec;
          asize = prec;
        }

        MPN_COPY(rp, up, asize);
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = u->_mp_exp;
        size = size >= 0 ? asize : -asize;



        MPF_ARRAY_SIZES(r.array)[r.idx] = size;
      }
    }

    namespace mpfRoutines
    {
      ANYCALLER void gpmpf_neg_mpf_array_idx(mpf_ptr r, mpf_array_idx u)
      {
        mp_size_t size;

        size = -MPF_ARRAY_SIZES(u.array)[u.idx];

        mp_size_t prec;
        mp_size_t asize;
        mp_ptr rp, up;

        prec = r->_mp_prec + 1; /* lie not to lose precision in assignment */
        asize = ABS(size);
        rp = r->_mp_d;
        up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);

        if (asize > prec)
        {
          up += asize - prec;
          asize = prec;
        }

        MPN_COPY(rp, up, asize);
        r->_mp_exp = MPF_ARRAY_EXPONENTS(u.array)[u.idx];
        size = size >= 0 ? asize : -asize;


        r->_mp_size = size;
      }
    }

  }
}