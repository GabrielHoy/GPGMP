#pragma once
#include "GPGMP/gpgmp.cuh"

namespace gpgmp {

    namespace host {

        //Copies all bytes of an arrayFrom to an arrayTo, utilizing the provided cudaMemcpyKind for the transfer direction.
        //arrayFrom and arrayTo may be on the host or device, but hostArray MUST be on the host and arrayFrom/arrayTo/hostArray MUST have equal metadata(precision, stored number count) in order for this function to operate correctly.
        //hostArray may be equal to arrayFrom or arrayTo! hostArray does not need to be a separate array but it must contain the same metadata as the arrays it is copying from/to.
        //This function is useful for copying an mpn_array struct from the host to the device or vice versa - or for simply copying the contents of an mpn_array.
        //Returns a cudaError_t error code associated with the memcpy operation.
        HOSTONLY static inline cudaError_t mpn_array_cudaMemcpy(mpn_array*& arrayTo, mpn_array*& arrayFrom, mpn_array*& hostArray, const cudaMemcpyKind memcpyKind) {
            return cudaMemcpy(
                arrayTo,
                arrayFrom,
                gpgmp::internal::mpn_array_get_struct_allocation_size(hostArray),
                memcpyKind
            );
        }

        //PROBABLY NOT THE FUNCTION YOU WANT!!!
        //This function performs (numIntsStored * 2) cudaMemcpy operations to only copy the data from the stored *number data* inside of arrayFrom to arrayTo. It is almost ALWAYS more efficient to use mpn_array_cudaMemcpy instead, which performs a single cudaMemcpy operation to copy the entire struct's memory - this should not be an issue if the array precision/count is the same between arrayFrom and arrayTo(as they should be...)
        //
        //Copies the contents of one mpn_array to another mpn_array, utilizing the provided cudaMemcpyKind for the transfer direction.
        //hostArray is a reference to an mpn_array we should use which has metadata(precision, count) to be used for the arrayFrom and arrayTo arrays. hostArray can be equal to arrayFrom or arrayTo, but it must be on the host. This is done since we don't know which array out of arrayFrom/arrayTo is on the host/device(if any).
        //Assumes that arrayTo and arrayFrom have the EXACT SAME PROPERTIES(precision, count) as hostArray.
        //Do not use this function to copy an mpn_array into a new, unallocated mpn_array. Both arrays must already be allocated, lest you face undefined behavior and segfaults.
        //Returns a cudaError_t error code associated with the memcpy operation.
        HOSTONLY static inline cudaError_t mpn_array_cudaMemcpy_dataOnly(mpn_array*& arrayTo, mpn_array*& arrayFrom, mpn_array*& hostArray, const cudaMemcpyKind memcpyKind) {
            const unsigned long long bytesFromArrayPtrToDataArrayPtr = ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpn_array));
            const unsigned long long bytesFromArrayPtrToSizesArrayPtr = bytesFromArrayPtrToDataArrayPtr + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * hostArray->numLimbsPerInteger * hostArray->numIntegersInArray);

            const mp_limb_t* arrayFromData = reinterpret_cast<const mp_limb_t*>(reinterpret_cast<const char*>(arrayFrom) + bytesFromArrayPtrToDataArrayPtr);
            const int* arrayFromSizes = reinterpret_cast<const int*>(reinterpret_cast<const char*>(arrayFrom) + bytesFromArrayPtrToSizesArrayPtr);

            mp_limb_t* arrayToData = reinterpret_cast<mp_limb_t*>(reinterpret_cast<char*>(arrayTo) + bytesFromArrayPtrToDataArrayPtr);
            int* arrayToSizes = reinterpret_cast<int*>(reinterpret_cast<char*>(arrayTo) + bytesFromArrayPtrToSizesArrayPtr);

            const int limbCountToCopyPerInt = hostArray->numLimbsPerInteger;
            cudaError_t err;

            //Copy the data from the source array to the destination array.
            // **OPTIMIZATION FOR LATER** - Revamp to use cudaMemcpyBatchAsync? I'd imagine it is wildly more performant.
            for (int idxToCopy = 0; idxToCopy < hostArray->numIntegersInArray; idxToCopy++) {
                //Copy the limb data from the source array to the destination array.
                err = cudaMemcpy(
                    arrayToData + (idxToCopy * limbCountToCopyPerInt),
                    arrayFromData + (idxToCopy * limbCountToCopyPerInt),
                    limbCountToCopyPerInt * sizeof(mp_limb_t),
                    memcpyKind
                );
                if (err != cudaSuccess) {
                    return err;
                }
                //Then copy the limb count used from the source array to the destination array.
                err = cudaMemcpy(
                    arrayToSizes + idxToCopy,
                    arrayFromSizes + idxToCopy,
                    sizeof(int),
                    memcpyKind
                );
                if (err != cudaSuccess) {
                    return err;
                }
            }

            return err;
        }

        //PROBABLY NOT THE FUNCTION YOU WANT!!!
        //This function performs (numIntsStored * 2) cudaMemcpy operations to only copy the data from the stored *number data* inside of arrayFrom to arrayTo. It is almost ALWAYS more efficient to use mpn_array_cudaMemcpy instead, which performs a single cudaMemcpy operation to copy the entire struct's memory - this should not be an issue if the array precision/count is the same between arrayFrom and arrayTo(as they should be...)
        //
        //Copies the contents of one mpn_array to another mpn_array, utilizing the provided cudaMemcpyKind for the transfer direction.
        //Performs many cudaMemcpy operations to only copy the data from the stored *number data* inside of arrayFrom to arrayTo. It is almost ALWAYS more efficient to use mpn_array_cudaMemcpy instead, which performs a single cudaMemcpy operation to copy the entire struct's memory - this should not be an issue if the array precision/count is the same between arrayFrom and arrayTo(as they should be...)
        //hostArray is a reference to an mpn_array we should use which has metadata(precision, count) to be used for the arrayFrom and arrayTo arrays. hostArray can be equal to arrayFrom or arrayTo, but it must be on the host. This is done since we don't know which array out of arrayFrom/arrayTo is on the host/device(if any).
        //Assumes that arrayTo and arrayFrom have the EXACT SAME PROPERTIES(precision, count) as hostArray.
        //Do not use this function to copy an mpn_array into a new, unallocated mpn_array. Both arrays must already be allocated, lest you face undefined behavior and segfaults.
        //NO ATTEMPT AT ERROR CHECKING IS DONE FOR THE SAKE OF SPEED; THERE IS NO WAY TO VALIDATE WHETHER ALL MEMCPY OPERATIONS PERFORMED BY THIS FUNCTION WERE SUCCESSFUL OR NOT.
        HOSTONLY static inline void mpn_array_cudaMemcpy_dataOnly_unsafe(mpn_array*& arrayTo, mpn_array*& arrayFrom, mpn_array*& hostArray, const cudaMemcpyKind memcpyKind) {
            const unsigned long long bytesFromArrayPtrToDataArrayPtr = ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpn_array));
            const unsigned long long bytesFromArrayPtrToSizesArrayPtr = bytesFromArrayPtrToDataArrayPtr + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * hostArray->numLimbsPerInteger * hostArray->numIntegersInArray);

            const mp_limb_t* arrayFromData = reinterpret_cast<const mp_limb_t*>(reinterpret_cast<const char*>(arrayFrom) + bytesFromArrayPtrToDataArrayPtr);
            const int* arrayFromSizes = reinterpret_cast<const int*>(reinterpret_cast<const char*>(arrayFrom) + bytesFromArrayPtrToSizesArrayPtr);

            mp_limb_t* arrayToData = reinterpret_cast<mp_limb_t*>(reinterpret_cast<char*>(arrayTo) + bytesFromArrayPtrToDataArrayPtr);
            int* arrayToSizes = reinterpret_cast<int*>(reinterpret_cast<char*>(arrayTo) + bytesFromArrayPtrToSizesArrayPtr);

            const int limbCountToCopyPerInt = hostArray->numLimbsPerInteger;

            //Copy the data from the source array to the destination array.
            // **OPTIMIZATION FOR LATER** - Revamp to use cudaMemcpyBatchAsync; I'd imagine it is wildly more performant.
            for (int idxToCopy = 0; idxToCopy < hostArray->numIntegersInArray; idxToCopy++) {
                //Copy the limb data from the source array to the destination array.
                cudaMemcpy(
                    arrayToData + (idxToCopy * limbCountToCopyPerInt),
                    arrayFromData + (idxToCopy * limbCountToCopyPerInt),
                    limbCountToCopyPerInt * sizeof(mp_limb_t),
                    memcpyKind
                );
                //Then copy the limb count used from the source array to the destination array.
                cudaMemcpy(
                    arrayToSizes + idxToCopy,
                    arrayFromSizes + idxToCopy,
                    sizeof(int),
                    memcpyKind
                );
            }
        }

    }
}