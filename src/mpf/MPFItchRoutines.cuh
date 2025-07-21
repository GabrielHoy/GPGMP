#pragma once

#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpfRoutines
  {
    ANYCALLER static int gpmpf_add_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
      return maxPrecisionLimbCountOfOperands;
    }


    ANYCALLER static int gpmpf_sub_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
      return maxPrecisionLimbCountOfOperands + 1;
    }


    ANYCALLER static int gpmpf_div_itch(mpf_ptr r, mpf_srcptr numerator, mpf_srcptr denominator)
    {
        return (PREC(r) * 2) + 1 + gpgmp::mpnRoutines::gpmpn_div_q_itch_intermediary(PREC(numerator), PREC(denominator));
    }
    ANYCALLER static int gpmpf_div_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
        return (maxPrecisionLimbCountOfOperands * 2) + 1 + gpgmp::mpnRoutines::gpmpn_div_q_itch_intermediary_maximum(maxPrecisionLimbCountOfOperands);
    }

    ANYCALLER static int gpmpf_ui_div_itch(mpf_ptr r, unsigned long int u, mpf_srcptr v)
    {
      return ABSIZ(v) + (1 + ((PREC(r) + 1) - (1 - (ABSIZ(v)) + 1))) + (PTR(r) == PTR(v) ? ABSIZ(v) : 0) + gpgmp::mpnRoutines::gpmpn_tdiv_qr_itch(ABSIZ(r), ABSIZ(v));
    }
    ANYCALLER static int gpmpf_ui_div_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
        return maxPrecisionLimbCountOfOperands +
        (1 +
          (
            (maxPrecisionLimbCountOfOperands + 1) -
            (1 - (maxPrecisionLimbCountOfOperands) + 1)
          )
        ) +
        maxPrecisionLimbCountOfOperands +
        gpgmp::mpnRoutines::gpmpn_tdiv_qr_itch(maxPrecisionLimbCountOfOperands);
    }


    ANYCALLER static int gpmpf_div_ui_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
        return maxPrecisionLimbCountOfOperands + 1;
    }


    ANYCALLER static int gpmpf_mul_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
        return maxPrecisionLimbCountOfOperands * 2;
    }


    ANYCALLER static int gpmpf_reldiff_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
        return (maxPrecisionLimbCountOfOperands * 2) + 1;
    }


    ANYCALLER static int gpmpf_sqrt_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
        return (maxPrecisionLimbCountOfOperands * 2) + gpgmp::mpnRoutines::gpmpn_sqrtrem_itch(maxPrecisionLimbCountOfOperands);
    }


    ANYCALLER static int gpmpf_sqrt_ui_itch(mp_size_t maxPrecisionLimbCountOfOperands)
    {
        mp_size_t rsize, zeros;
        zeros = 2 * maxPrecisionLimbCountOfOperands - 2;
        rsize = zeros + 1 + (GMP_NUMB_BITS < BITS_PER_ULONG);

        return (2 * maxPrecisionLimbCountOfOperands - 2) +
        1 +
        (GMP_NUMB_BITS < BITS_PER_ULONG)
        + gpgmp::mpnRoutines::gpmpn_sqrtrem_itch(rsize);
    }


    ANYCALLER static int gpmpf_set_q_itch(mp_size_t r_mp_prec, mpq_t q)
    {
      mp_size_t nsize, dsize, qsize, prospective_qsize, tsize, zeros;

      nsize = SIZ(&q->_mp_num);
      dsize = SIZ(&q->_mp_den);

      nsize = ABS(nsize);

      prospective_qsize = nsize - dsize + 1;
      qsize = r_mp_prec + 1;

      zeros = qsize - prospective_qsize;
      tsize = nsize + zeros;

      return (tsize + 1) + gpgmp::mpnRoutines::gpmpn_div_q_itch_intermediary(tsize, dsize);
    }
    ANYCALLER static int gpmpf_set_q_itch(mp_size_t r_mp_prec, mp_size_t q_mp_num_mp_size, mp_size_t q_mp_den_mp_size)
    {
      mp_size_t nsize, dsize, qsize, prospective_qsize, tsize, zeros;

      nsize = q_mp_num_mp_size;
      dsize = q_mp_den_mp_size;

      nsize = ABS(nsize);

      prospective_qsize = nsize - dsize + 1;
      qsize = r_mp_prec + 1;

      zeros = qsize - prospective_qsize;
      tsize = nsize + zeros;

      return (tsize + 1) + gpgmp::mpnRoutines::gpmpn_div_q_itch_intermediary(tsize, dsize);
    }


    ANYCALLER static int gpmpf_pow_ui_itch()
    {
      return -1337; //TODO: Implement me
    }
  }

  namespace mpfArrayRoutines
  {
    //This returns a rough maximum value for the scratch space required - in bytes - for the given input parameters.
    //This will more than likely be a huge number, be careful.
    //It is INCREDIBLY recommended to simply use the gpmpf_set_str routine on the host-side instead of device-side if you can, that way you don't need to allocate any scratch space yourself.
    ANYCALLER static size_t gpmpf_set_str_itch(mpf_array_idx x, size_t strSizeSettingFrom, int base)
    {
      mp_size_t ma, prec;
      prec = x.array->userSpecifiedPrecisionLimbCount + 1;
      LIMBS_PER_DIGIT_IN_BASE(ma, strSizeSettingFrom, base);

      return (strSizeSettingFrom + 1) + ((ma * sizeof(mp_limb_t))) + ((((2 * (prec + 1)) * sizeof(mp_limb_t)) * 2)) + (((prec + 1) * sizeof(mp_limb_t))) + (((prec + 1) * sizeof(mp_limb_t))) + (((prec * 2) * sizeof(mp_limb_t)));
    }
  }
}
