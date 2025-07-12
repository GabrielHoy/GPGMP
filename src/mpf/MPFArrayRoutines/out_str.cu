#define _GNU_SOURCE /* for DECIMAL_POINT in langinfo.h */

#include "config.cuh"

#include <stdio.h>
#include <string.h>

#if HAVE_LANGINFO_H
#include <langinfo.h> /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h> /* for localeconv */
#endif

#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    //Host-only due to the use of dynamic memory allocation,
    //as well as there generally not being a device-side equivalent of stream I/O(to my knowledge...)
    HOSTONLY size_t gpmpf_out_str(FILE *stream, int base, size_t n_digits, mpf_array_idx op)
    {
      char *str;
      mp_exp_t exp;
      size_t written;
      TMP_DECL;

      TMP_MARK;

      if (base == 0)
        base = 10;
      if (n_digits == 0)
        MPF_SIGNIFICANT_DIGITS(n_digits, base, op.array->userSpecifiedPrecisionLimbCount);

      if (stream == 0)
        stream = stdout;

      str = (char *)TMP_ALLOC(n_digits + 2); /* extra for minus sign and \0 */

      gpmpf_get_str(str, &exp, base, n_digits, op);
      n_digits = strlen(str);

      written = 0;

      /* Write sign */
      if (str[0] == '-')
      {
        str++;
        fputc('-', stream);
        written = 1;
        n_digits--;
      }

      {
        const char *point = GMP_DECIMAL_POINT;
        size_t pointlen = strlen(point);
        putc('0', stream);
        fwrite(point, 1, pointlen, stream);
        written += pointlen + 1;
      }

      /* Write mantissa */
      {
        size_t fwret;
        fwret = fwrite(str, 1, n_digits, stream);
        written += fwret;
      }

      /* Write exponent */
      {
        int fpret;
        fpret = fprintf(stream, (base <= 10 ? "e%ld" : "@%ld"), exp);
        written += fpret;
      }

      TMP_FREE;
      return ferror(stream) ? 0 : written;
    }

  }
}