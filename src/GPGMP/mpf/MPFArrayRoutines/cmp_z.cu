#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER int
    gpmpf_cmp_z(mpf_array_idx u, mpz_srcptr v) __GMP_NOTHROW
    {
      mpf_t vf;
      mp_size_t size;

      SIZ(vf) = size = SIZ(v);
      EXP(vf) = size = ABS(size);
      PTR(vf) = PTR(v);

      return gpgmp::internal::mpfArrayRoutines::gpmpf_cmp_array_idx_to_mpf_t(u, vf);
    }

  }
}