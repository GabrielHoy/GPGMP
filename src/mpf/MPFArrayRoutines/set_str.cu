/*
  This still needs work, as suggested by some FIXME comments.
  1. Don't depend on superfluous mantissa digits.
  2. Allocate temp space more cleverly.
  3. Use mpn_div_q instead of mpn_lshift+mpn_divrem.
*/

#define _GNU_SOURCE /* for DECIMAL_POINT in langinfo.h */

#include "config.cuh"

#include "DeviceCommon.cuh"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if HAVE_LANGINFO_H
#include <langinfo.h> /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h> /* for localeconv */
#endif

#include "gpgmp-impl.cuh"
#include "longlong.cuh"

#define digit_value_tab __gmp_digit_value_tab

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* Compute base^exp and return the most significant prec limbs in rp[].
       Put the count of omitted low limbs in *ign.
       Return the actual size (which might be less than prec).  */
    ANYCALLER static mp_size_t
    mpn_pow_1_highpart(mp_ptr rp, mp_size_t *ignp,
                       mp_limb_t base, mp_exp_t exp,
                       mp_size_t prec, mp_ptr tp)
    {
      mp_size_t ign; /* counts number of ignored low limbs in r */
      mp_size_t off; /* keeps track of offset where value starts */
      mp_ptr passed_rp = rp;
      mp_size_t rn;
      int cnt;
      int i;

      rp[0] = base;
      rn = 1;
      off = 0;
      ign = 0;
      count_leading_zeros(cnt, exp);
      for (i = GMP_LIMB_BITS - cnt - 2; i >= 0; i--)
      {
        gpgmp::mpnRoutines::gpmpn_sqr(tp, rp + off, rn);
        rn = 2 * rn;
        rn -= tp[rn - 1] == 0;
        ign <<= 1;

        off = 0;
        if (rn > prec)
        {
          ign += rn - prec;
          off = rn - prec;
          rn = prec;
        }
        MP_PTR_SWAP_DATA(rp, tp, prec);

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
        ign += rn - prec;
        rp += rn - prec;
        rn = prec;
      }

      MPN_COPY_INCR(passed_rp, rp + off, rn);
      *ignp = ign;
      return rn;
    }

    //Requires scratch space to be provided by the caller if called on the device-side.
    //Requires scratch space NOT to be provided if called on the host-side.
    //Using this routine on the device is HEAVILY RECOMMENDED AGAINST, albeit possible.
    ANYCALLER int
    gpmpf_set_str(mpf_array_idx x, const char *str, int base, char* scratchSpaceIfOnDevice)
    {
      size_t str_size;
      #ifdef __CUDA_ARCH__
        ASSERT(scratchSpaceIfOnDevice != nullptr);
        size_t scratchSpaceConsumedSoFar;
      #else
        ASSERT(scratchSpaceIfOnDevice == nullptr);
      #endif
      char *s, *begs;
      size_t i, j;
      int c;
      int negative;
      char *dotpos;
      const char *expptr;
      int exp_base;
      const char *point = GMP_DECIMAL_POINT;
      size_t pointlen = gpgmp::internal::cudaStrLen(point);
      const unsigned char *digit_value;
      int incr;
      size_t n_zeros_skipped;

      #ifndef __CUDA_ARCH__
        TMP_DECL;
      #endif

      c = (unsigned char)*str;

      /* Skip whitespace.  */
      while (gpgmp::internal::cudaIsSpace(c))
        c = (unsigned char)*++str;

      negative = 0;
      if (c == '-')
      {
        negative = 1;
        c = (unsigned char)*++str;
      }

      /* Default base to decimal.  */
      if (base == 0)
        base = 10;

      exp_base = base;

      if (base < 0)
      {
        exp_base = 10;
        base = -base;
      }

      digit_value = digit_value_tab;
      if (base > 36)
      {
        /* For bases > 36, use the collating sequence
     0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.  */
        digit_value += 208;
        if (base > 62)
          return -1; /* too large base */
      }

      /* Require at least one digit, possibly after an initial decimal point.  */
      if (digit_value[c] >= base)
      {
        /* not a digit, must be a decimal point */
        for (i = 0; i < pointlen; i++)
          if (str[i] != point[i])
            return -1;
        if (digit_value[(unsigned char)str[pointlen]] >= base)
          return -1;
      }

      /* Locate exponent part of the input.  Look from the right of the string,
         since the exponent is usually a lot shorter than the mantissa.  */
      expptr = NULL;
      str_size = gpgmp::internal::cudaStrLen(str);
      for (i = str_size - 1; i > 0; i--)
      {
        c = (unsigned char)str[i];
        if (c == '@' || (base <= 10 && (c == 'e' || c == 'E')))
        {
          expptr = str + i + 1;
          str_size = i;
          break;
        }
      }

      #ifdef __CUDA_ARCH__
        s = begs = scratchSpaceIfOnDevice;//(char *)TMP_ALLOC(str_size + 1);
        scratchSpaceConsumedSoFar = str_size + 1;
      #else
        TMP_MARK;
        s = begs = (char *)TMP_ALLOC(str_size + 1);
      #endif

      incr = 0;
      n_zeros_skipped = 0;
      dotpos = NULL;

      /* Loop through mantissa, converting it from ASCII to raw byte values.  */
      for (i = 0; i < str_size; i++)
      {
        c = (unsigned char)*str;
        if (!gpgmp::internal::cudaIsSpace(c))
        {
          int dig;

          for (j = 0; j < pointlen; j++)
            if (str[j] != point[j])
              goto not_point;
          if (1)
          {
            if (dotpos != 0)
            {
              /* already saw a decimal point, another is invalid */
              #ifndef __CUDA_ARCH__
              TMP_FREE;
              #endif
              return -1;
            }
            dotpos = s;
            str += pointlen - 1;
            i += pointlen - 1;
          }
          else
          {
          not_point:
            dig = digit_value[c];
            if (dig >= base)
            {
              return -1;
            }
            *s = dig;
            incr |= dig != 0;
            s += incr; /* Increment after first non-0 digit seen. */
            if (dotpos != NULL)
              /* Count skipped zeros between radix point and first non-0
                 digit. */
              n_zeros_skipped += 1 - incr;
          }
        }
        c = (unsigned char)*++str;
      }

      str_size = s - begs;

      {
        long exp_in_base;
        mp_size_t ra, ma, rn, mn;
        int cnt;
        mp_ptr mp, tp, rp;
        mp_exp_t exp_in_limbs;
        mp_size_t prec = x.array->userSpecifiedPrecisionLimbCount + 1;
        int divflag;
        mp_size_t madj, radj;

        if (str_size == 0)
        {
          MPF_ARRAY_SIZES(x.array)[x.idx] = 0;
          MPF_ARRAY_EXPONENTS(x.array)[x.idx] = 0;
          return 0;
        }

        LIMBS_PER_DIGIT_IN_BASE(ma, str_size, base);
        #ifdef __CUDA_ARCH__
          mp = reinterpret_cast<mp_ptr>(scratchSpaceIfOnDevice + scratchSpaceConsumedSoFar);
          scratchSpaceConsumedSoFar += (ma * sizeof(mp_limb_t));
        #else
          mp = TMP_ALLOC_LIMBS(ma);
        #endif
        mn = gpgmp::mpnRoutines::gpmpn_set_str(mp, (unsigned char *)begs, str_size, base);

        madj = 0;
        /* Ignore excess limbs in MP,MSIZE.  */
        if (mn > prec)
        {
          madj = mn - prec;
          mp += mn - prec;
          mn = prec;
        }

        if (expptr != 0)
        {
          /* Scan and convert the exponent, in base exp_base.  */
          long dig, minus, plusminus;
          c = (unsigned char)*expptr;
          minus = -(long)(c == '-');
          plusminus = minus | -(long)(c == '+');
          expptr -= plusminus; /* conditional increment */
          c = (unsigned char)*expptr++;
          dig = digit_value[c];
          if (dig >= exp_base)
          {
            #ifndef __CUDA_ARCH__
              TMP_FREE;
            #endif
            return -1;
          }
          exp_in_base = dig;
          c = (unsigned char)*expptr++;
          dig = digit_value[c];
          while (dig < exp_base)
          {
            exp_in_base = exp_in_base * exp_base;
            exp_in_base += dig;
            c = (unsigned char)*expptr++;
            dig = digit_value[c];
          }
          exp_in_base = (exp_in_base ^ minus) - minus; /* conditional negation */
        }
        else
          exp_in_base = 0;
        if (dotpos != 0)
          exp_in_base -= s - dotpos + n_zeros_skipped;
        divflag = exp_in_base < 0;
        exp_in_base = ABS(exp_in_base);

        if (exp_in_base == 0)
        {
          MPN_COPY(MPF_ARRAY_DATA_AT_IDX(x.array, x.idx), mp, mn);
          MPF_ARRAY_SIZES(x.array)[x.idx] = negative ? -mn : mn;
          MPF_ARRAY_EXPONENTS(x.array)[x.idx] = mn + madj;
          #ifndef __CUDA_ARCH__
            TMP_FREE;
          #endif
          return 0;
        }

        ra = 2 * (prec + 1);
        #ifdef __CUDA_ARCH__
          rp = reinterpret_cast<mp_ptr>(scratchSpaceIfOnDevice + scratchSpaceConsumedSoFar);
          scratchSpaceConsumedSoFar += ra * sizeof(mp_limb_t);
          tp = reinterpret_cast<mp_ptr>(scratchSpaceIfOnDevice + scratchSpaceConsumedSoFar);
          scratchSpaceConsumedSoFar += ra * sizeof(mp_limb_t);
        #else
          TMP_ALLOC_LIMBS_2(rp, ra, tp, ra);
        #endif
        rn = mpn_pow_1_highpart(rp, &radj, (mp_limb_t)base, exp_in_base, prec, tp);

        if (divflag)
        {

          mp_ptr qp;
          mp_limb_t qlimb;
          if (mn < rn)
          {
            /* Pad out MP,MSIZE for current divrem semantics.  */
            #ifdef __CUDA_ARCH__
              mp_ptr tmp = reinterpret_cast<mp_ptr>(scratchSpaceIfOnDevice + scratchSpaceConsumedSoFar);
              scratchSpaceConsumedSoFar += (rn + 1) * sizeof(mp_limb_t);
            #else
              mp_ptr tmp = TMP_ALLOC_LIMBS(rn + 1);
            #endif
            MPN_ZERO(tmp, rn - mn);
            MPN_COPY(tmp + rn - mn, mp, mn);
            mp = tmp;
            madj -= rn - mn;
            mn = rn;
          }
          if ((rp[rn - 1] & GMP_NUMB_HIGHBIT) == 0)
          {
            mp_limb_t cy;
            count_leading_zeros(cnt, rp[rn - 1]);
            cnt -= GMP_NAIL_BITS;
            gpgmp::mpnRoutines::gpmpn_lshift(rp, rp, rn, cnt);
            cy = gpgmp::mpnRoutines::gpmpn_lshift(mp, mp, mn, cnt);
            if (cy)
              mp[mn++] = cy;
          }

          #ifdef __CUDA_ARCH__
            qp = reinterpret_cast<mp_ptr>(scratchSpaceIfOnDevice + scratchSpaceConsumedSoFar);
            scratchSpaceConsumedSoFar += (prec + 1) * sizeof(mp_limb_t);
          #else
            qp = TMP_ALLOC_LIMBS(prec + 1);
          #endif
          qlimb = gpgmp::mpnRoutines::gpmpn_divrem(qp, prec - (mn - rn), mp, mn, rp, rn);
          tp = qp;
          exp_in_limbs = qlimb + (mn - rn) + (madj - radj);
          rn = prec;
          if (qlimb != 0)
          {
            tp[prec] = qlimb;
            /* Skip the least significant limb not to overrun the destination
               variable.  */
            tp++;
          }

        }
        else
        {
          #ifdef __CUDA_ARCH__
            tp = reinterpret_cast<mp_ptr>(scratchSpaceIfOnDevice + scratchSpaceConsumedSoFar);
            scratchSpaceConsumedSoFar += (rn + mn) * sizeof(mp_limb_t);
          #else
            tp = TMP_ALLOC_LIMBS(rn + mn);
          #endif
          if (rn > mn)
            gpgmp::mpnRoutines::gpmpn_mul(tp, rp, rn, mp, mn);
          else
            gpgmp::mpnRoutines::gpmpn_mul(tp, mp, mn, rp, rn);
          rn += mn;
          rn -= tp[rn - 1] == 0;
          exp_in_limbs = rn + madj + radj;

          if (rn > prec)
          {
            tp += rn - prec;
            rn = prec;
            exp_in_limbs += 0;
          }
        }

        MPN_COPY(MPF_ARRAY_DATA_AT_IDX(x.array, x.idx), tp, rn);
        MPF_ARRAY_SIZES(x.array)[x.idx] = negative ? -rn : rn;
        MPF_ARRAY_EXPONENTS(x.array)[x.idx] = exp_in_limbs;
        #ifndef __CUDA_ARCH__
          TMP_FREE;
        #endif
        return 0;
      }
    }

  }
}