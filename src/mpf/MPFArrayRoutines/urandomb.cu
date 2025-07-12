#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void gpmpf_urandomb(mpf_array_idx rop, gmp_randstate_ptr rstate, mp_bitcnt_t nbits)
    {
      mp_ptr rp;
      mp_size_t nlimbs;
      mp_exp_t exp;
      mp_size_t prec;

      rp = MPF_ARRAY_DATA_AT_IDX(rop.array, rop.idx);
      nlimbs = BITS_TO_LIMBS(nbits);
      prec = rop.array->userSpecifiedPrecisionLimbCount;

      if (nlimbs > prec + 1 || nlimbs == 0)
      {
        nlimbs = prec + 1;
        nbits = nlimbs * GMP_NUMB_BITS;
      }

      _gmp_rand(rp, rstate, nbits);

      /* If nbits isn't a multiple of GMP_NUMB_BITS, shift up.  */
      if (nbits % GMP_NUMB_BITS != 0)
        gpgmp::mpnRoutines::gpmpn_lshift(rp, rp, nlimbs, GMP_NUMB_BITS - nbits % GMP_NUMB_BITS);

      exp = 0;
      while (nlimbs != 0 && rp[nlimbs - 1] == 0)
      {
        nlimbs--;
        exp--;
      }
      MPF_ARRAY_EXPONENTS(rop.array)[rop.idx] = exp;
      MPF_ARRAY_SIZES(rop.array)[rop.idx] = nlimbs;
    }

  }
}