#pragma once
#include "gpgmp.cuh"

namespace gpgmp {

    namespace host {

        //Copies all bytes of an arrayFrom to an arrayTo, utilizing the provided cudaMemcpyKind for the transfer direction.
        //arrayFrom and arrayTo may be on the host or device, but hostArray MUST be on the host and arrayFrom/arrayTo/hostArray MUST have equal metadata(precision, stored number count) in order for this function to operate correctly.
        //hostArray may be equal to arrayFrom or arrayTo! hostArray does not need to be a separate array but it must contain the same metadata as the arrays it is copying from/to.
        //This function is useful for copying an mpf_array struct from the host to the device or vice versa - or for simply copying the contents of an mpf_array.
        //Returns a cudaError_t error code associated with the memcpy operation.
        HOSTONLY inline cudaError_t mpf_array_cudaMemcpy(mpf_array*& arrayTo, mpf_array*& arrayFrom, mpf_array*& hostArray, const cudaMemcpyKind memcpyKind) {
            return cudaMemcpy(
                arrayTo,
                arrayFrom,
                gpgmp::internal::mpf_array_get_struct_allocation_size(hostArray),
                memcpyKind
            );
        }

    }
}