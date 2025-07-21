#include <stdio.h>
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    //Does NOT free the string's data, unlike the original GMP implementation!
    HOSTONLY void gpmpf_dump(mpf_array_idx u)
    {
      mp_exp_t exp;
      char *str;

      str = gpmpf_get_str(0, &exp, 10, 0, u);
      if (str[0] == '-')
        printf("-0.%se%ld\n", str + 1, exp);
      else
        printf("0.%se%ld\n", str, exp);
    }

  }
}