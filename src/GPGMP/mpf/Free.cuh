#pragma once
#include "GPGMP/gpgmp.cuh"

namespace gpgmp {

    namespace host {
        //Frees an mpf_host_array struct from CPU memory.
        HOSTONLY void mpf_array_free_host(mpf_host_array array) {
            free(array);
        }

        //Frees an mpf_device_array struct from GPU memory, asynchronously if a stream is provided.
        HOSTONLY cudaError_t mpf_array_free_device(mpf_device_array array, cudaStream_t stream = 0) {
            if (stream) {
                return cudaFreeAsync(array, stream);
            } else {
                return cudaFree(array);
            }
        }
    }
}