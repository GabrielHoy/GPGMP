#pragma once
#include "gpgmp.cuh"

namespace gpgmp {

    namespace internal {

        //Returns the total size needed to pass to malloc() in order to allocate a contiguous block of memory large enough
        //to store an entire gpgmp_mpf_array struct along with the arrays it 'contains'.
        //This will likely be a bit larger than the 'necessary' size due to the struct and its sub-arrays each being aligned to 128-byte boundaries(thus the malloc'ed size for them will each be rounded up to a multiple of 128).
        ANYCALLER static inline size_t mpf_array_get_struct_allocation_size(const int arraySize, const mp_bitcnt_t precision) {
            return ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mpf_array)) //Struct members
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * arraySize) //mp_sizes array
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * arraySize) //mp_exp array
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * (LIMB_COUNT_FROM_PRECISION_BITS(precision)+1) * arraySize); //mp_array_data array - add +1 to precision to preserve accuracy in future calculations
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