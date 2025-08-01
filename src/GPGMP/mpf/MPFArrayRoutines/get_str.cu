#include <stdlib.h> /* for NULL */
#include "GPGMP/gpgmp-impl.cuh"
#include "GPGMP/longlong.cuh" /* for count_leading_zeros */

namespace gpgmp
{
  namespace mpfArrayRoutines
  {
    ANYCALLER static mp_size_t
    mpn_pow_1_highpart(mp_ptr rp, mp_size_t *ignp,
                       mp_limb_t base, unsigned long exp,
                       mp_size_t prec, mp_ptr scratchSpace)
    {
      mp_size_t ign; /* counts number of ignored low limbs in r */
      mp_size_t off; /* keeps track of offset where value starts */
      mp_ptr passed_rp = rp;
      mp_size_t rn;
      int cnt;
      int i;

      if (exp == 0)
      {
        rp[0] = 1;
        *ignp = 0;
        return 1;
      }

      rp[0] = base;
      rn = 1;
      off = 0;
      ign = 0;
      count_leading_zeros(cnt, exp);
      for (i = GMP_LIMB_BITS - cnt - 2; i >= 0; i--)
      {
        gpgmp::mpnRoutines::gpmpn_sqr(scratchSpace, rp + off, rn);
        rn = 2 * rn;
        rn -= scratchSpace[rn - 1] == 0;
        ign <<= 1;

        off = 0;
        if (rn > prec)
        {
          ign += rn - prec;
          off = rn - prec;
          rn = prec;
        }
        MP_PTR_SWAP_DATA(rp, scratchSpace, prec);
        //MP_PTR_SWAP(rp, scratchSpace);

        if (((exp >> i) & 1) != 0)
        {
          mp_limb_t cy;
          cy = gpgmp::mpnRoutines::gpmpn_mul_1(rp, rp + off, rn, base);
          rp[rn] = cy;
          rn += cy != 0;
          off = 0;
        }
      }

      if (rn > prec)
      {
        ASSERT(rn == prec + 1);

        ign += rn - prec;
        rp += rn - prec;
        rn = prec;
      }

      /* With somewhat less than 50% probability, we can skip this copy.  */
      if (passed_rp != rp + off)
        MPN_COPY_INCR(passed_rp, rp + off, rn);
      *ignp = ign;
      return rn;
    }

    //Host-only due to the use of dynamic memory allocation,
    //I may come back to this in the future to allow device-side string conversion, but I've decided against it for now - for the sake of time.
    HOSTONLY char* gpmpf_get_str(char *dbuf, mp_exp_t *exp, int base, size_t n_digits, mpf_array_idx u)
    {
      mp_exp_t ue;
      mp_size_t n_limbs_needed;
      size_t max_digits;
      mp_ptr up, pp, tp;
      mp_size_t un, pn, tn;
      unsigned char *tstr;
      mp_exp_t exp_in_base;
      size_t n_digits_computed;
      mp_size_t i;
      const char *num_to_text;
      size_t alloc_size = 0;
      char *dp;
      TMP_DECL;

      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
      un = ABS(MPF_ARRAY_SIZES(u.array)[u.idx]);
      ue = MPF_ARRAY_EXPONENTS(u.array)[u.idx];

      num_to_text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
      if (base > 1)
      {
        if (base <= 36)
          num_to_text = "0123456789abcdefghijklmnopqrstuvwxyz";
        else if (UNLIKELY(base > 62))
          return NULL;
      }
      else if (base > -2)
      {
        base = 10;
      }
      else
      {
        base = -base;
        if (UNLIKELY(base > 36))
          return NULL;
      }

      MPF_SIGNIFICANT_DIGITS(max_digits, base, u.array->userSpecifiedPrecisionLimbCount);
      if (n_digits == 0 || n_digits > max_digits)
        n_digits = max_digits;

      if (dbuf == 0)
      {
        /* We didn't get a string from the user.  Allocate one (and return
     a pointer to it) with space for `-' and terminating null.  */
        alloc_size = n_digits + 2;
        dbuf = __GMP_ALLOCATE_FUNC_TYPE(n_digits + 2, char);
      }

      if (un == 0)
      {
        *exp = 0;
        *dbuf = 0;
        n_digits = 0;
        goto done;
      }

      TMP_MARK;

      /* Allocate temporary digit space.  We can't put digits directly in the user
         area, since we generate more digits than requested.  (We allocate
         2 * GMP_LIMB_BITS extra bytes because of the digit block nature of the
         conversion.)  */
      tstr = (unsigned char *)TMP_ALLOC(n_digits + 2 * GMP_LIMB_BITS + 3);

      LIMBS_PER_DIGIT_IN_BASE(n_limbs_needed, n_digits, base);

      if (un > n_limbs_needed)
      {
        up += un - n_limbs_needed;
        un = n_limbs_needed;
      }

      TMP_ALLOC_LIMBS_2(pp, 2 * n_limbs_needed + 4,
                        tp, 2 * n_limbs_needed + 4);

      if (ue <= n_limbs_needed)
      {
        /* We need to multiply number by base^n to get an n_digits integer part.  */
        mp_size_t n_more_limbs_needed, ign, off;
        unsigned long e;

        n_more_limbs_needed = n_limbs_needed - ue;
        DIGITS_IN_BASE_PER_LIMB(e, n_more_limbs_needed, base);

        pn = mpn_pow_1_highpart(pp, &ign, (mp_limb_t)base, e, n_limbs_needed + 1, tp);
        if (un > pn)
          gpgmp::mpnRoutines::gpmpn_mul(tp, up, un, pp, pn); /* FIXME: mpn_mul_highpart */
        else
          gpgmp::mpnRoutines::gpmpn_mul(tp, pp, pn, up, un); /* FIXME: mpn_mul_highpart */
        tn = un + pn;
        tn -= tp[tn - 1] == 0;
        off = un - ue - ign;
        if (off < 0)
        {
          MPN_COPY_DECR(tp - off, tp, tn);
          MPN_ZERO(tp, -off);
          tn -= off;
          off = 0;
        }
        n_digits_computed = gpgmp::mpnRoutines::gpmpn_get_str(tstr, base, tp + off, tn - off);

        exp_in_base = n_digits_computed - e;
      }
      else
      {
        /* We need to divide number by base^n to get an n_digits integer part.  */
        mp_size_t n_less_limbs_needed, ign, off, xn;
        unsigned long e;
        mp_ptr dummyp, xp;

        n_less_limbs_needed = ue - n_limbs_needed;
        DIGITS_IN_BASE_PER_LIMB(e, n_less_limbs_needed, base);

        pn = mpn_pow_1_highpart(pp, &ign, (mp_limb_t)base, e, n_limbs_needed + 1, tp);

        xn = n_limbs_needed + (n_less_limbs_needed - ign);
        xp = TMP_ALLOC_LIMBS(xn);
        off = xn - un;
        MPN_ZERO(xp, off);
        MPN_COPY(xp + off, up, un);

        dummyp = TMP_ALLOC_LIMBS(pn);
        mp_limb_t* scratchForTDivQR = TMP_ALLOC_LIMBS(gpgmp::mpnRoutines::gpmpn_tdiv_qr_itch(xn, pn));
        gpgmp::mpnRoutines::gpmpn_tdiv_qr(tp, dummyp, (mp_size_t)0, xp, xn, pp, pn, scratchForTDivQR);
        tn = xn - pn + 1;
        tn -= tp[tn - 1] == 0;
        n_digits_computed = gpgmp::mpnRoutines::gpmpn_get_str(tstr, base, tp, tn);

        exp_in_base = n_digits_computed + e;
      }

      /* We should normally have computed too many digits.  Round the result
         at the point indicated by n_digits.  */
      if (n_digits_computed > n_digits)
      {
        size_t i;
        /* Round the result.  */
        if (tstr[n_digits] * 2 >= base)
        {
          n_digits_computed = n_digits;
          for (i = n_digits - 1;; i--)
          {
            unsigned int x;
            x = ++(tstr[i]);
            if (x != base)
              break;
            n_digits_computed--;
            if (i == 0)
            {
              /* We had something like `bbbbbbb...bd', where 2*d >= base
                 and `b' denotes digit with significance base - 1.
                 This rounds up to `1', increasing the exponent.  */
              tstr[0] = 1;
              n_digits_computed = 1;
              exp_in_base++;
              break;
            }
          }
        }
      }

      /* We might have fewer digits than requested as a result of rounding above,
         (i.e. 0.999999 => 1.0) or because we have a number that simply doesn't
         need many digits in this base (e.g., 0.125 in base 10).  */
      if (n_digits > n_digits_computed)
        n_digits = n_digits_computed;

      /* Remove trailing 0.  There can be many zeros.  */
      while (n_digits != 0 && tstr[n_digits - 1] == 0)
        n_digits--;

      dp = dbuf + (MPF_ARRAY_SIZES(u.array)[u.idx] < 0);

      /* Translate to ASCII and copy to result string.  */
      for (i = 0; i < n_digits; i++)
        dp[i] = num_to_text[tstr[i]];
      dp[n_digits] = 0;

      *exp = exp_in_base;

      if (MPF_ARRAY_SIZES(u.array)[u.idx] < 0)
      {
        dbuf[0] = '-';
        n_digits++;
      }

      TMP_FREE;

    done:
      /* If the string was alloced then resize it down to the actual space
         required.  */
      if (alloc_size != 0)
      {
        __GMP_REALLOCATE_FUNC_MAYBE_TYPE(dbuf, alloc_size, n_digits + 1, char);
      }

      return dbuf;
    }

  }
}