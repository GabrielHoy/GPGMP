#pragma once

#include "gpgmp-impl.cuh"

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

  }
}
