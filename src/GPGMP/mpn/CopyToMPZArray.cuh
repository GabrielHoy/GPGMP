#pragma once
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp {

    namespace host {

        //Copies the contents of an mpn_array to an mpz_t array.
        //Assumes that the given mpz_t array has enough array indices to store all integers from the given mpn_array.
        //Allocates new memory for integer limbs inside of the mpz_t array if necessary depending on the precision of the given mpn_array.
        //The mpz_t array given to this function MUST be initialized!.
        HOSTONLY static inline void mpn_array_copy_to_mpz_array(mpz_t* arrayCopyInto, mpn_host_array& arrayCopyFrom) {
            mp_limb_t* limbDataArray = MPN_ARRAY_DATA(arrayCopyFrom);
            int* sizesArray = MPN_ARRAY_SIZES(arrayCopyFrom);

            for (int idxToCopy = 0; idxToCopy < arrayCopyFrom->numIntegersInArray; idxToCopy++) {
                mpz_t& copyInto = arrayCopyInto[idxToCopy];
                const int limbCountToCopy = sizesArray[idxToCopy];
                mp_limb_t* limbDataToCopyInto = MPZ_REALLOC(copyInto, limbCountToCopy); //Reallocates the integer limbs in copyInto if necessary in order to fit the used limb data from the mpn_array integer at index i.

                SIZ(copyInto) = limbCountToCopy;
                memcpy(limbDataToCopyInto, limbDataArray + (idxToCopy * arrayCopyFrom->numLimbsPerInteger), limbCountToCopy * sizeof(mp_limb_t));
            }
        }

        //Copies the contents of an mpn_array to an mpz_t array, initializing each mpz_t in the given array with exactly enough precision to hold the data copied from the given mpn_array.
        //Assumes that the given mpz_t array has enough array indices to store all integers from the given mpn_array.
        HOSTONLY static inline void mpn_array_copy_to_mpz_array_with_mpz_init(mpz_t* arrayCopyInto, mpn_host_array& arrayCopyFrom) {
            mp_limb_t* limbDataArray = MPN_ARRAY_DATA(arrayCopyFrom);
            int* sizesArray = MPN_ARRAY_SIZES(arrayCopyFrom);

            for (int idxToCopy = 0; idxToCopy < arrayCopyFrom->numIntegersInArray; idxToCopy++) {
                mpz_t& copyInto = arrayCopyInto[idxToCopy];
                const int limbCountToCopy = sizesArray[idxToCopy];

                mpz_init2(copyInto, PRECISION_BITS_FROM_LIMB_COUNT(limbCountToCopy)); //Initializes the mpz_t with exactly enough precision to hold the limb data we need to copy from arrayCopyFrom.

                SIZ(copyInto) = limbCountToCopy;
                memcpy(copyInto->_mp_d, limbDataArray + (idxToCopy * arrayCopyFrom->numLimbsPerInteger), limbCountToCopy * sizeof(mp_limb_t));
            }
        }

        //Copies the contents of an mpn_array from the device to an mpz_t array on the host.
        //Assumes that the given mpz_t array has enough array indices to store all integers from the given mpn_array.
        //Requires manually giving the precision and count of integers in the given mpn_array; this is due to the inability to query these values from the device pointer on the host.
        //Performs (arraySize * 2) cudaMemcpy operations for both the limb data and limbs used by the integers in the given mpn_array.
        //Returns a cudaError_t value indicating the success or failure of the copy operation(s).
        HOSTONLY static inline cudaError_t mpn_array_copy_to_mpz_array_from_device(mpz_t* arrayCopyInto, mpn_device_array& arrayCopyFrom, const int arraySize, const mp_bitcnt_t precision) {
            mp_limb_t* limbDataArray = MPN_ARRAY_DATA_NO_PTR_INDEXING(arrayCopyFrom);
            int* sizesArray = MPN_ARRAY_SIZES_NO_PTR_INDEXING(arrayCopyFrom, arraySize, precision);
            int numLimbsPerInteger = LIMB_COUNT_FROM_PRECISION_BITS(precision);

            cudaError_t err;

            for (int idxToCopy = 0; idxToCopy < arraySize; idxToCopy++) {
                mpz_t& copyInto = arrayCopyInto[idxToCopy];

                //We need to get the limb count of the current integer in the mpn_array, to do this we need to actually copy that data over from the device.
                int limbCountToCopy = 0;
                err = cudaMemcpy(
                    &limbCountToCopy,
                    sizesArray + idxToCopy,
                    sizeof(int),
                    cudaMemcpyDeviceToHost
                );
                if (err != cudaSuccess) {
                    return err;
                }

                mp_limb_t* mpzLimbDataToCopyInto = MPZ_REALLOC(copyInto, ABS(limbCountToCopy));

                //Now we can copy the limb data over from the device to the mpz_t.
                err = cudaMemcpy(
                    mpzLimbDataToCopyInto,
                    limbDataArray + (idxToCopy * numLimbsPerInteger),
                    ABS(limbCountToCopy) * sizeof(mp_limb_t),
                    cudaMemcpyDeviceToHost
                );
                if (err != cudaSuccess) {
                    return err;
                }

                //Finally, let's update the used limb count of the mpz_t to match the limb count of the integer in the mpn_array.
                SIZ(copyInto) = limbCountToCopy;
            }

            return err;
        }

        //Copies the contents of an mpn_array from the device to an mpz_t array on the host, initializing each mpz_t in the given array with exactly enough precision to hold the data copied from the given mpn_array.
        //Assumes that the given mpz_t array has enough array indices to store all integers from the given mpn_array.
        //Requires manually giving the precision and count of integers in the given mpn_array; this is due to the inability to query these values from the device pointer on the host.
        //Performs (arraySize * 2) cudaMemcpy operations for both the limb data and limbs used by the integers in the given mpn_array.
        //Returns a cudaError_t value indicating the success or failure of the copy operation(s).
        HOSTONLY static inline cudaError_t mpn_array_copy_to_mpz_array_from_device_with_mpz_init(mpz_t* arrayCopyInto, mpn_device_array& arrayCopyFrom, const int arraySize, const mp_bitcnt_t precision) {
            mp_limb_t* limbDataArray = MPN_ARRAY_DATA_NO_PTR_INDEXING(arrayCopyFrom);
            int* sizesArray = MPN_ARRAY_SIZES_NO_PTR_INDEXING(arrayCopyFrom, arraySize, precision);
            int numLimbsPerInteger = LIMB_COUNT_FROM_PRECISION_BITS(precision);

            cudaError_t err;

            for (int idxToCopy = 0; idxToCopy < arraySize; idxToCopy++) {
                mpz_t& copyInto = arrayCopyInto[idxToCopy];

                //We need to get the limb count of the current integer in the mpn_array, to do this we need to actually copy that data over from the device.
                int limbCountToCopy = 0;
                err = cudaMemcpy(
                    &limbCountToCopy,
                    sizesArray + idxToCopy,
                    sizeof(int),
                    cudaMemcpyDeviceToHost
                );
                if (err != cudaSuccess) {
                    return err;
                }

                mpz_init2(copyInto, PRECISION_BITS_FROM_LIMB_COUNT(ABS(limbCountToCopy))); //Initializes the mpz_t with exactly enough precision to hold the limb data we need to copy from arrayCopyFrom.

                //Now we can copy the limb data over from the device to the mpz_t.
                err = cudaMemcpy(
                    copyInto->_mp_d,
                    limbDataArray + (idxToCopy * numLimbsPerInteger),
                    ABS(limbCountToCopy) * sizeof(mp_limb_t),
                    cudaMemcpyDeviceToHost
                );
                if (err != cudaSuccess) {
                    return err;
                }

                //Finally, let's update the used limb count of the mpz_t to match the limb count of the integer in the mpn_array.
                SIZ(copyInto) = limbCountToCopy;
            }

            return err;
        }

    }

}