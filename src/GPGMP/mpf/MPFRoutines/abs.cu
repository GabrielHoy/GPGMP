/* mpf_abs -- Compute the absolute value of a float.

Copyright 1993-1995, 2001 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfRoutines
  {

    /* Compute the absolute value of a floating-point number.
       Parameters:
       - result: Pointer to the destination floating-point number
       - input: Pointer to the source floating-point number

       The function handles both in-place and separate destination cases.
       For in-place operation (result == input), it simply makes the size positive.
       For separate destination, it copies the data with precision handling. */
    ANYCALLER void
    gpmpf_abs(mpf_ptr result, mpf_srcptr input)
    {
      /* Get the absolute value of the input's size (number of limbs) */
      mp_size_t num_limbs = ABS(input->_mp_size);

      /* If result and input are different locations, we need to copy data */
      if (result != input)
      {
        mp_size_t precision;
        mp_ptr result_digits, input_digits;

        /* Add 1 to precision to avoid losing precision during assignment */
        precision = result->_mp_prec + 1;
        result_digits = result->_mp_d;
        input_digits = input->_mp_d;

        /* If input has more limbs than our precision, truncate it */
        if (num_limbs > precision)
        {
          /* Skip the most significant digits that exceed our precision */
          input_digits += num_limbs - precision;
          num_limbs = precision;
        }

        /* Copy the digits from input to result */
        MPN_COPY(result_digits, input_digits, num_limbs);
        /* Copy the exponent */
        result->_mp_exp = input->_mp_exp;
      }

      /* Set the result's size to be positive (absolute value) */
      result->_mp_size = num_limbs;
    }

  }
}