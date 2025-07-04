#pragma once
#include "gpgmp.cuh"

namespace gpgmp {

    namespace internal {

        //Returns the total number of limbs required to perform all given operations specified by a UsedOperationFlags enum and a given precision.
        //The returned value of this is meant to be added to allocation limb sizes when allocating space for new floating-point numbers, as "additional limbs" after the _mp_prec limbs allocated in the _mp_d field.
        //?Optimization Potential: fixme this is naive and awful...can be obviously optimized by a lot, but it's 8am and im still up
        ANYCALLER static inline int mpf_get_scratch_space_limb_count(const int precisionInLimbs, const _UsedOperationFlags operations) {
            int scratchSpaceLimbsRequired = 0;

            if (operations & OP_ADD) {
                scratchSpaceLimbsRequired = precisionInLimbs;
            }

            if (operations & OP_SUB || operations & OP_DIV_UI) { //both div_ui and sub need prec(r)+1 limbs for scratch space
                scratchSpaceLimbsRequired = precisionInLimbs + 1;
            }

            if (operations & OP_MUL || operations & OP_SQRT) {
                //OP_MUL: MAX(MAX(ABSIZ(u), prec) + MAX(ABSIZ(v), prec), 2 * ABSIZ(u))
                //OP_SQRT: 2 * (PREC(r)) - (EXP(u) & 1);
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, precisionInLimbs * 2);
            }

            if (operations & OP_RELDIFF || operations & OP_DIV) {
                //RELDIFF: PREC(rdiff) + ABSIZ(x) + 1
                //DIV: MAX((SIZ(u) - MAX(-(SIZ(u) - SIZ(V)), 0)) + (SIZ(u) - SIZ(v)) + 1, (SIZ(u) - MAX(-(SIZ(u) - SIZ(V)), 0)) + 1) + SIZ(v)
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, (precisionInLimbs * 2) + 1);
            }

            if (operations & OP_SQRT_UI) {
                //(2 * PREC(r) - 2) + 1 + U2
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, (2 * precisionInLimbs - 2) + 1 + (GMP_NUMB_BITS < BITS_PER_ULONG));
            }

            if (operations & OP_UI_DIV) {
                //ABSIZ(v) + (1 + ((PREC(r) + 1) - (1 - (ABSIZ(v)) + 1))) + (PTR(r) == PTR(v) ? ABSIZ(v) : 0)
                scratchSpaceLimbsRequired = MAX(scratchSpaceLimbsRequired, precisionInLimbs * 4);
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