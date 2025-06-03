#pragma once

#define WANT_ASSERT true

//Useful macros ported over from GMP
#define SGN(x)       ((x)<0 ? -1 : (x) != 0)
#define ABS(x)       ((x)>=0 ? (x) : -(x))
#undef MIN
#define MIN(l,o) ((l) < (o) ? (l) : (o))
#undef MAX
#define MAX(h,i) ((h) > (i) ? (h) : (i))

#define ALLOC(x) ((x)->_mp_alloc)
#define PTR(x) ((x)->_mp_d)
#define SIZ(x) ((x)->_mp_size)
#define ABSIZ(x) ABS (SIZ (x))


#define LIKELY(cond)                   __GMP_LIKELY(cond)
#define UNLIKELY(cond)                 __GMP_UNLIKELY(cond)

#define __GMPF_BITS_TO_PREC(n)						\
  ((mp_size_t) ((__GMP_MAX (53, n) + 2 * GMP_NUMB_BITS - 1) / GMP_NUMB_BITS))
#define __GMPF_PREC_TO_BITS(n) \
  ((mp_bitcnt_t) (n) * GMP_NUMB_BITS - GMP_NUMB_BITS)

/* Return non-zero if xp,xsize and yp,ysize overlap.
   If xp+xsize<=yp there's no overlap, or if yp+ysize<=xp there's no
   overlap.  If both these are false, there's an overlap. */
#define MPN_OVERLAP_P(xp, xsize, yp, ysize)				\
  ((xp) + (xsize) > (yp) && (yp) + (ysize) > (xp))
#define MEM_OVERLAP_P(xp, xsize, yp, ysize)				\
  (   (char *) (xp) + (xsize) > (char *) (yp)				\
   && (char *) (yp) + (ysize) > (char *) (xp))

/* Return non-zero if xp,xsize and yp,ysize are either identical or not
   overlapping.  Return zero if they're partially overlapping. */
#define MPN_SAME_OR_SEPARATE_P(xp, yp, size)				\
  MPN_SAME_OR_SEPARATE2_P(xp, size, yp, size)
#define MPN_SAME_OR_SEPARATE2_P(xp, xsize, yp, ysize)			\
  ((xp) == (yp) || ! MPN_OVERLAP_P (xp, xsize, yp, ysize))

  /* Return non-zero if dst,dsize and src,ssize are either identical or
   overlapping in a way suitable for an incrementing/decrementing algorithm.
   Return zero if they're partially overlapping in an unsuitable fashion. */
#define MPN_SAME_OR_INCR2_P(dst, dsize, src, ssize)			\
  ((dst) <= (src) || ! MPN_OVERLAP_P (dst, dsize, src, ssize))
#define MPN_SAME_OR_INCR_P(dst, src, size)				\
  MPN_SAME_OR_INCR2_P(dst, size, src, size)
#define MPN_SAME_OR_DECR2_P(dst, dsize, src, ssize)			\
  ((dst) >= (src) || ! MPN_OVERLAP_P (dst, dsize, src, ssize))
#define MPN_SAME_OR_DECR_P(dst, src, size)				\
  MPN_SAME_OR_DECR2_P(dst, size, src, size)



#if WANT_ASSERT
#include <intrin.h>

#ifndef __CUDA_ARCH__
#define ASSERT(expr) \
    if (!(expr)) { \
        printf("Assertion failed: %s\nBreaking for debugging...\n", #expr); \
        __debugbreak(); \
        exit(EXIT_FAILURE); \
    }
#else
#define ASSERT(expr) \
    if (!(expr)) { \
        printf("Assertion failed: %s\n...\n", #expr); \
    }
#endif

#else
#define ASSERT(expr) do {} while (0)
#endif


#define MPZ_REALLOC(z,n) (UNLIKELY ((n) > ALLOC(z))			\
			  ? (mp_ptr) _mpz_realloc(z,n)			\
			  : PTR(z))









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
