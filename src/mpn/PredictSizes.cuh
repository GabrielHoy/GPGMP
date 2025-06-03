#pragma once
#include "gpgmp.cuh"

namespace gpgmp {

    namespace internal {

        //Returns the total size needed to pass to malloc() in order to allocate a contiguous block of memory large enough
        //to store the entire gpgmp_mpn_array struct and the arrays it 'contains'.
        //This will likely be a bit larger than the 'necessary' size due to the struct and its sub-arrays each being aligned to 128-byte boundaries(and thus the malloc'ed size for them will be rounded up to a multiple of 128).
        ANYCALLER static inline size_t mpn_array_get_struct_allocation_size(const int arraySize, const mp_bitcnt_t precision) {
            mp_size_t precisionLimbCount = MPN_ARRAY_LIMB_COUNT_FROM_BITS(precision);
            return ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mpn_array)) //Struct members
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * precisionLimbCount * arraySize) //_mp_array_data array
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * arraySize); //_mp_sizes array
        }

        //Returns the total size needed to pass to malloc() in order to allocate a contiguous block of memory large enough
        //to store a copy of the given mpn_array.
        ANYCALLER static inline size_t mpn_array_get_struct_allocation_size(const mpn_array* array) {
            return ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mpn_array)) //Struct members
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * array->numLimbsPerInteger * array->numIntegersInArray) //_mp_array_data array
            + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * array->numIntegersInArray); //_mp_sizes array
        }

    }

}