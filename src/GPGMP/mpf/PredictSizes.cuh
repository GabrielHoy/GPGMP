#pragma once
#include "GPGMP/gpgmp.cuh"

namespace gpgmp {

    namespace internal {

        //Returns the total number of limbs required to perform all given operations specified by a UsedOperationFlags enum and a given precision.
        //The returned value of this is meant to be added to allocation limb sizes when allocating space for new floating-point numbers, as "additional limbs" after the _mp_prec limbs allocated in the _mp_d field.
        //?Optimization Potential: fixme this is naive and awful...can be obviously optimized by a lot, but it's 8am and im still up
        ANYCALLER static inline int mpf_get_scratch_space_limb_count(const int precisionInLimbs, const _UsedOperationFlags operations) {
            int scratchSpaceLimbsRequired = 0;

            if (operations & OP_ADD) {
                scratchSpaceLimbsRequired = gpgmp::mpfRoutines::gpmpf_add_itch(precisionInLimbs);
            }

            if (operations & OP_SUB) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_sub_itch(precisionInLimbs));
            }

            if (operations & OP_DIV_UI) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_div_ui_itch(precisionInLimbs));
            }

            if (operations & OP_MUL) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_mul_itch(precisionInLimbs));
            }

            if (operations & OP_SQRT) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_sqrt_itch(precisionInLimbs));
            }

            if (operations & OP_RELDIFF) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_reldiff_itch(precisionInLimbs));
            }

            if (operations & OP_DIV) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_div_itch(precisionInLimbs));
            }

            if (operations & OP_SQRT_UI) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_sqrt_ui_itch(precisionInLimbs));
            }

            if (operations & OP_UI_DIV) {
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, gpgmp::mpfRoutines::gpmpf_ui_div_itch(precisionInLimbs));
            }

            return scratchSpaceLimbsRequired;
        }

        //Returns the total size needed to pass to malloc() in order to allocate a contiguous block of memory large enough
        //to store an entire gpgmp_mpf_array struct along with the arrays it 'contains'.
        //This will likely be a bit larger than the 'necessary' size due to the struct and its sub-arrays each being aligned to 128-byte boundaries(thus the malloc'ed size for them will each be rounded up to a multiple of 128).
        ANYCALLER static inline size_t mpf_array_get_struct_allocation_size(const int arraySize, const mp_bitcnt_t precision, const _UsedOperationFlags operationsToMakeAvailable) {
            return ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mpf_array)) //Struct members
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * arraySize) //mp_sizes array
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * arraySize) //mp_exp array
            + ALIGN_TO_128_BYTE_MULTIPLE(
                sizeof(mp_limb_t) * //each limb byte size, with the following number of limbs...
                (
                    LIMB_COUNT_FROM_PRECISION_BITS(precision) + //the actual number of limbs needed to store the precision for the float...
                    1 + //...plus one for some extra precision...following GMP's lead...
                    mpf_get_scratch_space_limb_count(LIMB_COUNT_FROM_PRECISION_BITS(precision), operationsToMakeAvailable) //...plus the number of limbs required as scratch space when performing any given operations the user has specified they want to do.
                )
                * arraySize //...then that many limbs multiplied by the number of floats we want to store in the array.
            ); //mp_array_data array - add +1 to precision to preserve accuracy in future calculations, also add scratch space limbs.
        }

        //Returns the total size needed to pass to malloc() in order to allocate a contiguous block of memory large enough
        //to store a copy of the given gpgmp_mpf_array.
        ANYCALLER static inline size_t mpf_array_get_struct_allocation_size(const mpf_array* array) {
            return ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mpf_array)) //Struct members
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * array->numFloatsInArray) //mp_sizes array
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * array->numFloatsInArray) //mp_exp array
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * array->limbsPerArrayFloat * array->numFloatsInArray); //mp_array_data array
        }

    }

}