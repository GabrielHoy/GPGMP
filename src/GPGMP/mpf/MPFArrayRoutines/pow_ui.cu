#include "GPGMP/gpgmp-impl.cuh"
#include "GPGMP/longlong.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* This uses a plain right-to-left square-and-multiply algorithm.

       FIXME: When popcount(e) is not too small, it would probably speed things up
       to use a k-ary sliding window algorithm.  */

    ANYCALLER void gpmpf_pow_ui(mpf_array_idx r, mpf_array_idx b, unsigned long int e)
    {
      mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(r.array, r.idx);
      mpf_t t;
      int cnt;

      if (e <= 1)
      {
        if (e == 0)
        {
          gpmpf_set_ui(r, 1);
        }
        else
        {
          gpmpf_set(r, b);
        }
        return;
      }

      count_leading_zeros(cnt, (mp_limb_t)e);
      cnt = GMP_LIMB_BITS - 1 - cnt;

      /* Increase computation precision as a function of the exponent.  Adding
         log2(popcount(e) + log2(e)) bits should be sufficient, but we add log2(e),
         i.e. much more.  With mpf's rounding of precision to whole limbs, this
         will be excessive only when limbs are artificially small.  */
      t->_mp_size = 0;
      t->_mp_exp = 0;
      t->_mp_prec = r.array->userSpecifiedPrecisionLimbCount + cnt;
      t->_mp_d = scratchSpace;
      scratchSpace += t->_mp_prec;



      //gpmpf_set(t, b); /* consume most significant bit */
      gpgmp::internal::mpfRoutines::gpmpf_set_mpf_t_to_array_idx(t, b);

      while (--cnt > 0)
      {
        gpgmp::mpfRoutines::gpmpf_mul(t, t, t, scratchSpace);
        if ((e >> cnt) & 1)
          gpgmp::internal::mpfRoutines::gpmpf_mul_mpf_t_by_mpf_array_idx(t, t, b, scratchSpace);
      }

      /* Do the last iteration specially in order to save a copy operation.  */
      if (e & 1)
      {
        gpgmp::mpfRoutines::gpmpf_mul(t, t, t, scratchSpace);
        gpgmp::internal::mpfArrayRoutines::gpmpf_mul_mpf_t_by_mpf_array_idx(r, t, b, scratchSpace);
        //gpmpf_mul(r, t, b, scratchSpace);
      }
      else
      {
        gpgmp::internal::mpfArrayRoutines::gpmpf_mul_mpf_t_by_mpf_t(r, t, t, scratchSpace);
        //gpmpf_mul(r, t, t, scratchSpace);
      }

    }

  }
}