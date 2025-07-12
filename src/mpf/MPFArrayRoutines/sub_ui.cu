#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    ANYCALLER void
    gpmpf_sub_ui(mpf_array_idx sum, mpf_array_idx u, unsigned long int v)
    {
      __mpf_struct vv;
      mp_limb_t vl;

      if (v == 0)
      {
        gpmpf_set(sum, u);
        return;
      }

      vl = v;
      vv._mp_size = 1;
      vv._mp_d = &vl;
      vv._mp_exp = 1;
      gpgmp::internal::mpfArrayRoutines::gpmpf_sub_mpf_t_from_array_idx(sum, u, &vv);
    }

  }
}