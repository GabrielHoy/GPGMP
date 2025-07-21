#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* Compute the absolute value of a floating-point number.
       Parameters:
       - result: Pointer to the destination floating-point number
       - input: Pointer to the source floating-point number

       The function handles both in-place and separate destination cases.
       For in-place operation (result == input), it simply makes the size positive.
       For separate destination, it copies the data with precision handling. */
    ANYCALLER void gpmpf_abs(mpf_array_idx result, mpf_array_idx input)
    {
      mp_size_t num_limbs = MPF_ARRAY_SIZES(input.array)[input.idx];

      if ((result.array != input.array) || (result.idx != input.idx)) {
        mp_size_t precision;
        mp_ptr result_digits, input_digits;

        /* Add 1 to precision to avoid losing precision during assignment */
        precision = result.array->userSpecifiedPrecisionLimbCount+1;
        result_digits = MPF_ARRAY_DATA_AT_IDX(result.array, result.idx);
        input_digits = MPF_ARRAY_DATA_AT_IDX(input.array, input.idx);

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
        MPF_ARRAY_EXPONENTS(result.array)[result.idx] = MPF_ARRAY_EXPONENTS(input.array)[input.idx];
      }

      /* Set the result's size to be positive (absolute value) */
      MPF_ARRAY_SIZES(result.array)[result.idx] = num_limbs;
    }

  }
}