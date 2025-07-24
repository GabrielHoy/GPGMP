#pragma once
#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp {

    namespace host {
        //Frees an mpn_host_array struct from CPU memory.
        HOSTONLY static inline void mpn_array_free_host(mpn_host_array array)
        {
            free(array);
        }

        //Frees an mpn_device_array struct from GPU memory, asynchronously if a stream is provided.
        HOSTONLY static inline void mpn_array_free_device(mpn_device_array array, cudaStream_t stream = 0)
        {
            if (stream) {
                cudaFreeAsync(array, stream);
            } else {
                cudaFree(array);
            }
        }
    }

}