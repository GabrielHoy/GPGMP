#pragma once
#include "gpgmp.cuh"

namespace gpgmp {

    namespace host {

        //Allocates a new mpf_array struct on the host and assigns a provided pointer to it.
        //Returns true if the allocation was successful, false otherwise.
        HOSTONLY bool mpf_array_allocate_on_host(mpf_host_array& arrayPtr, const int numFloatsInArray, const mp_bitcnt_t precisionPerFloat) {
            const size_t totalSizeToAllocate = gpgmp::internal::mpf_array_get_struct_allocation_size(numFloatsInArray, precisionPerFloat);

            void* allocatedMemory = malloc(totalSizeToAllocate);
            if (allocatedMemory == NULL) {
                return false;
            }

            //Initialize the struct's metadata
            arrayPtr = reinterpret_cast<mpf_array*>(allocatedMemory);
            arrayPtr->numFloatsInArray = numFloatsInArray;
            arrayPtr->limbsPerArrayFloat = LIMB_COUNT_FROM_PRECISION_BITS(precisionPerFloat)+1;

            return true;
        }

        //Allocates enough space on the host inside of arrayPtr to store a direct copy of a given mpf_array matchSizeOf.
        //Returns true if the allocation was successful, false otherwise.
        HOSTONLY bool mpf_array_allocate_on_host(mpf_host_array& arrayPtr, mpf_host_array& matchSizeOf) {
            return mpf_array_allocate_on_host(arrayPtr, matchSizeOf->numFloatsInArray, PRECISION_BITS_FROM_LIMB_COUNT(matchSizeOf->limbsPerArrayFloat-1));
        }

        //Allocates a new mpf_array struct on the current CUDA device and assigns a provided pointer to it.
        //Returns the CUDA error code associated with the allocation attempt.
        HOSTONLY cudaError_t mpf_array_allocate_on_device(mpf_device_array& deviceArrayPtr, const int numFloatsInArray, const mp_bitcnt_t precisionPerFloat) {
            const size_t totalSizeToAllocate = gpgmp::internal::mpf_array_get_struct_allocation_size(numFloatsInArray, precisionPerFloat);

            cudaError_t err = cudaMalloc(&deviceArrayPtr, totalSizeToAllocate);
            if (err != cudaSuccess) {
                return err;
            }

            //I believe this can be optimized by directly using cudaMemset in order to set the struct fields to their appropriate values on the device using pointer offsets,
            //though for now this is more readable and easier to debug.
            mpf_array dataToMemcpyToDevice;
            dataToMemcpyToDevice.numFloatsInArray = numFloatsInArray;
            dataToMemcpyToDevice.limbsPerArrayFloat = LIMB_COUNT_FROM_PRECISION_BITS(precisionPerFloat)+1;

            err = cudaMemcpy(deviceArrayPtr, &dataToMemcpyToDevice, ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mpf_array)), cudaMemcpyHostToDevice);

            if (err != cudaSuccess) {
                //If we couldn't copy the struct data over to the device, free the memory we allocated for the struct and return the error code appropriately
                cudaFree(deviceArrayPtr);
            }

            return err;
        }

        //Allocates enough space on the device inside of arrayPtr to store a direct copy of a given mpf_array matchSizeOf.
        //matchSizeOf MUST be on the host.
        //Returns the CUDA error code associated with the allocation attempt.
        HOSTONLY cudaError_t mpf_array_allocate_on_device(mpf_device_array& arrayPtr, mpf_host_array& matchSizeOf) {
            return mpf_array_allocate_on_device(arrayPtr, matchSizeOf->numFloatsInArray, PRECISION_BITS_FROM_LIMB_COUNT(matchSizeOf->limbsPerArrayFloat-1));
        }
    }

}