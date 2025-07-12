#pragma once

#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpfRoutines
  {
    ANYCALLER static inline int mpf_add_itch(mpf_ptr r, mpf_srcptr u, mpf_srcptr v)
    {
      return PREC(r);
    }

    ANYCALLER static inline int mpf_sub_itch(mpf_ptr r, mpf_srcptr u, mpf_srcptr v)
    {
      return PREC(r) + 1;
    }

    ANYCALLER static inline int mpf_div_itch(mpf_ptr r, mpf_srcptr u, mpf_srcptr v)
    {
        return (PREC(r) * 2) + 1;
    }

    ANYCALLER static inline int mpf_ui_div_itch(mpf_ptr r, unsigned long int u, mpf_srcptr v)
    {
        return ABSIZ(v) + (1 + ((PREC(r) + 1) - (1 - (ABSIZ(v)) + 1))) + (PTR(r) == PTR(v) ? ABSIZ(v) : 0);
    }

    ANYCALLER static inline int mpf_div_ui_itch(mpf_ptr r, mpf_srcptr u, unsigned long int v)
    {
        return PREC(r) + 1;
    }

    ANYCALLER static inline int mpf_mul_itch(mpf_ptr r, mpf_srcptr u, mpf_srcptr v)
    {
        return PREC(r) * 2;
    }

    ANYCALLER static inline int mpf_reldiff_itch(mpf_ptr rdiff, mpf_srcptr x, mpf_srcptr y)
    {
        return (PREC(rdiff) * 2) + 1;
    }

    ANYCALLER static inline int mpf_sqrt_itch(mpf_ptr r, mpf_srcptr u)
    {
        return PREC(r) * 2;
    }

    ANYCALLER static inline int mpf_sqrt_ui_itch(mpf_ptr r, unsigned long int u)
    {
        return (2 * PREC(r) - 2) + 1 + (GMP_NUMB_BITS < BITS_PER_ULONG);
    }

    ANYCALLER static inline int mpf_pow_ui_itch()
    {
      return -1337; //TODO: Implement me
    }
  }

  namespace mpfArrayRoutines
  {
    //This returns a rough maximum value for the scratch space required - in bytes - for the given input parameters.
    //This will more than likely be a huge number, be careful.
    //It is INCREDIBLY recommended to simply use the gpmpf_set_str routine on the host-side instead of device-side if you can, that way you don't need to allocate any scratch space yourself.
    ANYCALLER static inline size_t gpmpf_set_str_itch(mpf_array_idx x, size_t strSizeSettingFrom, int base)
    {
      mp_size_t ma, prec;
      prec = x.array->userSpecifiedPrecisionLimbCount + 1;
      LIMBS_PER_DIGIT_IN_BASE(ma, strSizeSettingFrom, base);

      return (strSizeSettingFrom + 1) + ((ma * sizeof(mp_limb_t))) + ((((2 * (prec + 1)) * sizeof(mp_limb_t)) * 2)) + (((prec + 1) * sizeof(mp_limb_t))) + (((prec + 1) * sizeof(mp_limb_t))) + (((prec * 2) * sizeof(mp_limb_t)));
    }
  }
}
