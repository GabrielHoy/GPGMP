#pragma once

#define WANT_ASSERT true




//Useful macros specific to GPGMP
#define ANYCALLER __device__ __host__
#define HOSTONLY __host__
#define GPUONLY __device__
#define GPUKERNEL __global__

//Aligns a given byte count to the nearest 128-byte boundary for better memory coalescence.
#define ALIGN_TO_128_BYTE_MULTIPLE(byteSize) (((byteSize) + 127) & ~127)

//Returns the number of limbs needed to store a number with the given bit count.
#define MPN_ARRAY_LIMB_COUNT_FROM_BITS(bits) ((bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS)
//Returns the number of bits needed to store a number with the given limb count.
#define MPN_ARRAY_BITS_FROM_LIMB_COUNT(limbCount) (limbCount * GMP_LIMB_BITS)

//Array data is stored directly after the struct in mpn_array's, as one contiguous block of memory.
#define MPN_ARRAY_DATA(array) (reinterpret_cast<mp_limb_t*>(reinterpret_cast<char*>(array) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpn_array))))
//Array sizes are stored directly after the array data in mpn_array's, as one contiguous block of memory.
#define MPN_ARRAY_SIZES(array) (reinterpret_cast<int*>(reinterpret_cast<char*>(MPN_ARRAY_DATA(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * array->numLimbsPerInteger * array->numIntegersInArray)))
//MPN_ARRAY_DATA, but const.
#define MPN_ARRAY_DATA_CONST(array) (reinterpret_cast<const mp_limb_t*>(reinterpret_cast<const char*>(array) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpn_array))))
//MPN_ARRAY_SIZES, but const.
#define MPN_ARRAY_SIZES_CONST(array) (reinterpret_cast<const int*>(reinterpret_cast<const char*>(MPN_ARRAY_DATA_CONST(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * array->numLimbsPerInteger * array->numIntegersInArray)))
//Equivalent formula to MPN_ARRAY_DATA, without indexing any struct fields. This allows us to use this formula with device pointers on the host.
//This is literally just MPN_ARRAY_DATA but we define this macro for code cleanliness.
#define MPN_ARRAY_DATA_NO_PTR_INDEXING(array) MPN_ARRAY_DATA(array)
//Equivalent formula to MPN_ARRAY_SIZES, without indexing any struct fields. This allows us to use this formula with device pointers on the host.
#define MPN_ARRAY_SIZES_NO_PTR_INDEXING(array, arraySize, precision) (reinterpret_cast<int*>(reinterpret_cast<char*>(MPN_ARRAY_DATA_NO_PTR_INDEXING(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * MPN_ARRAY_LIMB_COUNT_FROM_BITS(precision) * arraySize)));
