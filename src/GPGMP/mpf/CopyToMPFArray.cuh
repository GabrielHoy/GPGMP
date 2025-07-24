#pragma once
#include "GPGMP/gpgmp.cuh"

namespace gpgmp {

    namespace host {

        //Copies the contents of an mpf_array to a "gmp-native" mpf_t array.
        //This is useful to transpose between gpgmp's mpf_array type and an array of mpf_t* used by GMP.
        //Assumes that the given mpf_t array has enough array indices to store all floats from the given mpf_array.
        //Assumes that the given mpf_t array has enough precision to store all floats from the given mpf_array.
        HOSTONLY static inline void mpf_array_copy_to_gmp_mpf_array(mpf_t* arrayCopyInto, mpf_host_array& arrayCopyFrom) {
            int* sizesArray = MPF_ARRAY_SIZES(arrayCopyFrom);
            mp_exp_t* exponentsArray = MPF_ARRAY_EXPONENTS(arrayCopyFrom);
            mp_limb_t* limbDataArrays = MPF_ARRAY_DATA(arrayCopyFrom);

            for (int idxToCopy = 0; idxToCopy < arrayCopyFrom->numFloatsInArray; idxToCopy++) {
                mpf_t& copyInto = arrayCopyInto[idxToCopy];
                //Quick check to ensure that the mpf_t copyInto has enough precision to hold the used limb data from the mpf_array index
                ASSERT(sizesArray[idxToCopy] <= PREC(copyInto));

                //Update the mpf_t's size and exponent to match the value in the mpf_array
                SIZ(copyInto) = sizesArray[idxToCopy];
                EXP(copyInto) = exponentsArray[idxToCopy];

                //Then copy the limb data from the mpf_array to the mpf_t
                memcpy(PTR(copyInto), limbDataArrays + (idxToCopy * arrayCopyFrom->limbsPerArrayFloat), sizesArray[idxToCopy] * sizeof(mp_limb_t));
            }
        }


        //Serves the same function as mpf_array_copy_to_gmp_mpf_array, but for uninitialized mpf_t arrays.
        //Copies the contents of an mpf_array to a "gmp-native" mpf_t array.
        //Initializes each mpf_t in the given mpf_t array with available precision equivalent to the mpf_array's.
        //This is useful to transpose between gpgmp's mpf_array type and an array of mpf_t* used by GMP.
        //Assumes that the given mpf_t array has enough array indices to store all floats from the given mpf_array.
        HOSTONLY static inline void mpf_array_copy_to_gmp_mpf_array_with_mpf_init(mpf_t* arrayCopyInto, mpf_host_array& arrayCopyFrom) {
            int* sizesArray = MPF_ARRAY_SIZES(arrayCopyFrom);
            mp_exp_t* exponentsArray = MPF_ARRAY_EXPONENTS(arrayCopyFrom);
            mp_limb_t* limbDataArrays = MPF_ARRAY_DATA(arrayCopyFrom);

            for (int idxToCopy = 0; idxToCopy < arrayCopyFrom->numFloatsInArray; idxToCopy++) {
                mpf_t& copyInto = arrayCopyInto[idxToCopy];

                int abSizForIdx = ABS(sizesArray[idxToCopy]);
                //copyInto at this point is assumed to be uninitialized - initialize it now with the precision we need to perform the copy operation.
                mpf_init2(copyInto, PRECISION_BITS_FROM_LIMB_COUNT(arrayCopyFrom->userSpecifiedPrecisionLimbCount));

                //Update the mpf_t's size and exponent to match the value in the mpf_array
                SIZ(copyInto) = sizesArray[idxToCopy];
                EXP(copyInto) = exponentsArray[idxToCopy];

                //Then copy the limb data from the mpf_array to the mpf_t
                memcpy(
                    PTR(copyInto),
                    limbDataArrays + (idxToCopy * arrayCopyFrom->limbsPerArrayFloat),
                    abSizForIdx * sizeof(mp_limb_t)
                );
            }
        }
    }

}