#pragma once
#include "GPGMP/gpgmp.cuh"

namespace gpgmp {

    namespace host {

        //Allocates a new mpn_array struct on the host and assigns a provided pointer to it.
        //Returns true if the allocation was successful, false otherwise.
        HOSTONLY static inline bool mpn_array_allocate_on_host(mpn_host_array& arrayPtr, const int arraySize, const mp_bitcnt_t precision) {
            const size_t sizeToAllocateForStruct = gpgmp::internal::mpn_array_get_struct_allocation_size(arraySize, precision);

            void* allocatedMemory = malloc(sizeToAllocateForStruct);
            if (allocatedMemory == NULL) {
                return false;
            }

            // Initialize the struct with the newly allocated memory
            arrayPtr = reinterpret_cast<mpn_array*>(allocatedMemory);
            arrayPtr->numIntegersInArray = arraySize;
            arrayPtr->numLimbsPerInteger = LIMB_COUNT_FROM_PRECISION_BITS(precision);

            return true;
        }

        //Allocates enough space on the host inside of arrayPtr to store a direct copy of a given mpn_array matchSizeOf.
        //Returns true if the allocation was successful, false otherwise.
        HOSTONLY static inline bool mpn_array_allocate_on_host(mpn_host_array& arrayPtr, mpn_host_array& matchSizeOf) {
            return mpn_array_allocate_on_host(arrayPtr, matchSizeOf->numIntegersInArray, PRECISION_BITS_FROM_LIMB_COUNT(matchSizeOf->numLimbsPerInteger));
        }


        //Allocates a new mpn_array struct on the current CUDA device and assigns a provided pointer to it.
        //Returns the CUDA error code associated with the allocation attempt.
        HOSTONLY static inline cudaError_t mpn_array_allocate_on_device(mpn_device_array& deviceArrayPtr, const int arraySize, const mp_bitcnt_t precision) {
            const size_t sizeToAllocateForStruct = gpgmp::internal::mpn_array_get_struct_allocation_size(arraySize, precision);

            cudaError_t err = cudaMalloc(&deviceArrayPtr, sizeToAllocateForStruct);
            if (err != cudaSuccess) {
                return err;
            }

            //I believe this can be optimized by directly using cudaMemset in order to set the struct fields to their appropriate values on the device using pointer offsets,
            //though for now this is more readable and easier to debug.
            mpn_array dataToMemcpyToDevice;
            dataToMemcpyToDevice.numIntegersInArray = arraySize;
            dataToMemcpyToDevice.numLimbsPerInteger = LIMB_COUNT_FROM_PRECISION_BITS(precision);

            err = cudaMemcpy(deviceArrayPtr, &dataToMemcpyToDevice, ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mpn_array)), cudaMemcpyHostToDevice);

            if (err != cudaSuccess) {
                //If we couldn't copy the struct data over to the device, free the memory we allocated for the struct and return the error code appropriately
                cudaFree(deviceArrayPtr);
            }


            return err;
        }

        //Allocates enough space on the device inside of arrayPtr to store a direct copy of a given mpn_array matchSizeOf.
        //matchSizeOf MUST be on the host.
        //Returns the CUDA error code associated with the allocation attempt.
        HOSTONLY static inline cudaError_t mpn_array_allocate_on_device(mpn_device_array& arrayPtr, mpn_host_array& matchSizeOf) {
            return mpn_array_allocate_on_device(arrayPtr, matchSizeOf->numIntegersInArray, PRECISION_BITS_FROM_LIMB_COUNT(matchSizeOf->numLimbsPerInteger));
        }

    }

}