#include <stdio.h>
#include <ctype.h>
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    //Host-only due to the use of dynamic memory allocation,
    //as well as there generally not being a device-side equivalent of stream I/O(to my knowledge...)
    HOSTONLY size_t gpmpf_inp_str(mpf_array_idx rop, FILE *stream, int base)
    {
      char *str;
      size_t alloc_size, str_size;
      int c;
      int res;
      size_t nread;

      if (stream == 0)
        stream = stdin;

      alloc_size = 100;
      str = __GMP_ALLOCATE_FUNC_TYPE(alloc_size, char);
      str_size = 0;
      nread = 0;

      /* Skip whitespace.  */
      do
      {
        c = getc(stream);
        nread++;
      } while (isspace(c));

      for (;;)
      {
        if (str_size >= alloc_size)
        {
          size_t old_alloc_size = alloc_size;
          alloc_size = alloc_size * 3 / 2;
          str = __GMP_REALLOCATE_FUNC_TYPE(str, old_alloc_size, alloc_size, char);
        }
        if (c == EOF || isspace(c))
          break;
        str[str_size++] = c;
        c = getc(stream);
      }
      ungetc(c, stream);
      nread--;

      if (str_size >= alloc_size)
      {
        size_t old_alloc_size = alloc_size;
        alloc_size = alloc_size * 3 / 2;
        str = __GMP_REALLOCATE_FUNC_TYPE(str, old_alloc_size, alloc_size, char);
      }
      str[str_size] = 0;

      res = gpmpf_set_str(rop, str, base, nullptr);
      (*__gpgmp_free_func)(str, alloc_size);

      if (res == -1)
        return 0; /* error */

      return str_size + nread;
    }

  }
}