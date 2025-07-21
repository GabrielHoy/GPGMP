#pragma once

#define WANT_ASSERT true

//Useful macros specific to GPGMP
#define ANYCALLER __device__ __host__
#define HOSTONLY __host__
#define GPUONLY __device__
#define GPUKERNEL __global__

//Aligns a given byte count to the nearest 128-byte boundary for better memory coalescence.
#define ALIGN_TO_128_BYTE_MULTIPLE(byteSize) (((byteSize) + 127) & ~127)
//Generic form of above formula that I will in all likelihoods never end up actually using.
#define ALIGN_TO_X_BYTE_MULTIPLE(byteSize, alignment) (((byteSize) + (alignment - 1)) & ~(alignment - 1))

//Returns the number of limbs needed to store a number with the given bit count.
#define LIMB_COUNT_FROM_PRECISION_BITS(bits) ((bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS)
//Returns the number of bits needed to store a number with the given limb count.
#define PRECISION_BITS_FROM_LIMB_COUNT(limbCount) (limbCount * GMP_LIMB_BITS)


/*\ MPN \*/

//Array data is stored directly after the struct in mpn_array's, as one contiguous block of memory.
#define MPN_ARRAY_DATA(array) (reinterpret_cast<mp_limb_t*>(reinterpret_cast<char*>(array) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpn_array))))
//Array sizes are stored directly after the array data in mpn_array's, as one contiguous block of memory.
#define MPN_ARRAY_SIZES(array) (reinterpret_cast<int*>(reinterpret_cast<char*>(MPN_ARRAY_DATA(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * array->numLimbsPerInteger * array->numIntegersInArray)))

//MPN_ARRAY_DATA, but const.
#define MPN_ARRAY_DATA_CONST(array) (reinterpret_cast<const mp_limb_t*>(reinterpret_cast<const char*>(array) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpn_array))))
//MPN_ARRAY_SIZES, but const.
#define MPN_ARRAY_SIZES_CONST(array) (reinterpret_cast<const int*>(reinterpret_cast<const char*>(MPN_ARRAY_DATA_CONST(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * array->numLimbsPerInteger * array->numIntegersInArray)))

//Equivalent formulas to MPN_ARRAY_DATA/MPN_ARRAY_SIZES, without indexing any struct fields. This allows us to use these formulas with device pointers on the host.
//Getting the array data specifically is literally just MPN_ARRAY_DATA but we define this macro for code cleanliness.
#define MPN_ARRAY_DATA_NO_PTR_INDEXING(array) MPN_ARRAY_DATA(array)
//Equivalent formula to MPN_ARRAY_SIZES, without indexing any struct fields. This allows us to use this formula with device pointers on the host.
#define MPN_ARRAY_SIZES_NO_PTR_INDEXING(array, arraySize, precision) (reinterpret_cast<int*>(reinterpret_cast<char*>(MPN_ARRAY_DATA_NO_PTR_INDEXING(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * LIMB_COUNT_FROM_PRECISION_BITS(precision) * arraySize)));

/*\ MPN \*/



/*\ MPF \*/

//Array sizes are stored directly after the struct in mpf_array's, as one contiguous block of memory.
#define MPF_ARRAY_SIZES(array) (reinterpret_cast<int*>(reinterpret_cast<char*>(array) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpf_array))))
//Exponents are stored directly after the array sizes in mpf_array's, as one contiguous block of memory.
#define MPF_ARRAY_EXPONENTS(array) (reinterpret_cast<mp_exp_t*>(reinterpret_cast<char*>(MPF_ARRAY_SIZES(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * array->numFloatsInArray)))
//Array data is stored directly after exponent data in mpf_array's, as one contiguous block of memory.
#define MPF_ARRAY_DATA(array) (reinterpret_cast<mp_limb_t*>(reinterpret_cast<char*>(MPF_ARRAY_EXPONENTS(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * array->numFloatsInArray)))
//Array scratch space is stored directly after the array data per-index.
#define MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(array, idx) (MPF_ARRAY_DATA_AT_IDX(array, idx) + (array->userSpecifiedPrecisionLimbCount+1))

//Helper macro to get a pointer to a number's index inside of an mpf_array, accounting for the stride between numbers.
#define MPF_ARRAY_DATA_AT_IDX(array, idx) (MPF_ARRAY_DATA(array) + (idx * array->limbsPerArrayFloat))

//Used to ensure that scratch space is available for a certain operation.
#define MPF_ARRAY_ASSERT_OP_AVAILABLE(array, op) ASSERT(array->availableOperations & op)

//MPF_ARRAY_SIZES, but const.
#define MPF_ARRAY_SIZES_CONST(array) (reinterpret_cast<const int*>(reinterpret_cast<const char*>(array) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(gpgmp::mpf_array))))
//MPF_ARRAY_EXPONENTS, but const.
#define MPF_ARRAY_EXPONENTS_CONST(array) (reinterpret_cast<const mp_exp_t*>(reinterpret_cast<const char*>(MPF_ARRAY_SIZES_CONST(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * array->numFloatsInArray)))
//MPF_ARRAY_DATA, but const.
#define MPF_ARRAY_DATA_CONST(array) (reinterpret_cast<const mp_limb_t*>(reinterpret_cast<const char*>(MPF_ARRAY_EXPONENTS_CONST(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * array->numFloatsInArray)))

//Equivalent formulas to MPF_ARRAY_SIZES/MPF_ARRAY_EXPONENTS/MPF_ARRAY_DATA, without indexing any struct fields. This allows us to use these formulas with device pointers on the host.
//Getting the array sizes specifically is literally just MPF_ARRAY_SIZES but we define this macro for code cleanliness.
#define MPF_ARRAY_SIZES_NO_PTR_INDEXING(array) MPF_ARRAY_SIZES(array)
//Equivalent formula to MPF_ARRAY_EXPONENTS, without indexing any struct fields. This allows us to use this formula with device pointers on the host.
#define MPF_ARRAY_EXPONENTS_NO_PTR_INDEXING(array, arraySize) (reinterpret_cast<mp_exp_t*>(reinterpret_cast<char*>(MPF_ARRAY_SIZES(array)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * arraySize)))
//Equivalent formula to MPF_ARRAY_DATA, without indexing any struct fields. This allows us to use this formula with device pointers on the host.
#define MPF_ARRAY_DATA_NO_PTR_INDEXING(array, arraySize) (reinterpret_cast<mp_limb_t*>(reinterpret_cast<char*>(MPF_ARRAY_EXPONENTS_NO_PTR_INDEXING(array, arraySize)) + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * arraySize)))

/*\ MPF \*/