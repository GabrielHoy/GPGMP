#pragma once

#define WANT_ASSERT true




//Useful macros specific to GPGMP
#define ANYCALLER __device__ __host__
#define HOSTONLY __host__
#define GPUONLY __device__
#define GPUKERNEL __global__

//GPGMP's equivalent of __GMP_DECLSPEC.
#define __GPGMP_DECLSPEC
#define __GPGMP_MPN(x) x

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









/* BEGIN MPN INTERNALS PORTING VOMIT */

//Useful macros ported over from GMP
#define SGN(x)       ((x)<0 ? -1 : (x) != 0)
#define ABS(x)       ((x)>=0 ? (x) : -(x))
#define NEG_CAST(T,x) (- (__GMP_CAST (T, (x) + 1) - 1))
#define ABS_CAST(T,x) ((x) >= 0 ? __GMP_CAST (T, x) : NEG_CAST (T,x))
#undef MIN
#define MIN(l,o) ((l) < (o) ? (l) : (o))
#undef MAX
#define MAX(h,i) ((h) > (i) ? (h) : (i))

#define ALLOC(x) ((x)->_mp_alloc)
#define PTR(x) ((x)->_mp_d)
#define SIZ(x) ((x)->_mp_size)
#define ABSIZ(x) ABS (SIZ (x))

#define MPN_CMP(result, xp, yp, size)  __GMPN_CMP(result, xp, yp, size)
#define LIKELY(cond)                   __GMP_LIKELY(cond)
#define UNLIKELY(cond)                 __GMP_UNLIKELY(cond)

#define __GMPF_BITS_TO_PREC(n)						\
  ((mp_size_t) ((__GMP_MAX (53, n) + 2 * GMP_NUMB_BITS - 1) / GMP_NUMB_BITS))
#define __GMPF_PREC_TO_BITS(n) \
  ((mp_bitcnt_t) (n) * GMP_NUMB_BITS - GMP_NUMB_BITS)

#if defined _LONG_LONG_LIMB
#define CNST_LIMB(C) ((mp_limb_t) C##LL)
#else /* not _LONG_LONG_LIMB */
#define CNST_LIMB(C) ((mp_limb_t) C##L)
#endif /* _LONG_LONG_LIMB */

#define MPN_FIB2_SIZE(n) \
  ((mp_size_t) ((n) / 32 * 23 / GMP_NUMB_BITS) + 4)

/* Macros for altering parameter order according to regparm usage. */
#if USE_LEADING_REGPARM
#define REGPARM_2_1(a,b,x)    x,a,b
#define REGPARM_3_1(a,b,c,x)  x,a,b,c
#define REGPARM_ATTR(n) __attribute__ ((regparm (n)))
#else
#define REGPARM_2_1(a,b,x)    a,b,x
#define REGPARM_3_1(a,b,c,x)  a,b,c,x
#define REGPARM_ATTR(n)
#endif


/* "const" basically means a function does nothing but examine its arguments
   and give a return value, it doesn't read or write any memory (neither
   global nor pointed to by arguments), and has no other side-effects.  This
   is more restrictive than "pure".  See info node "(gcc)Function
   Attributes".  __GMP_NO_ATTRIBUTE_CONST_PURE lets tune/common.c etc turn
   this off when trying to write timing loops.  */
#if HAVE_ATTRIBUTE_CONST && ! defined (__GMP_NO_ATTRIBUTE_CONST_PURE)
#define ATTRIBUTE_CONST  __attribute__ ((const))
#else
#define ATTRIBUTE_CONST
#endif



#if GMP_NUMB_BITS != 64
Error, error, this data is for 64 bits
#endif

#define FIB_TABLE_LIMIT         93
#define FIB_TABLE_LUCNUM_LIMIT  92

const mp_limb_t
__gmp_fib_table[FIB_TABLE_LIMIT+2] = {
  CNST_LIMB (0x1),  /* -1 */
  CNST_LIMB (0x0),  /* 0 */
  CNST_LIMB (0x1),  /* 1 */
  CNST_LIMB (0x1),  /* 2 */
  CNST_LIMB (0x2),  /* 3 */
  CNST_LIMB (0x3),  /* 4 */
  CNST_LIMB (0x5),  /* 5 */
  CNST_LIMB (0x8),  /* 6 */
  CNST_LIMB (0xd),  /* 7 */
  CNST_LIMB (0x15),  /* 8 */
  CNST_LIMB (0x22),  /* 9 */
  CNST_LIMB (0x37),  /* 10 */
  CNST_LIMB (0x59),  /* 11 */
  CNST_LIMB (0x90),  /* 12 */
  CNST_LIMB (0xe9),  /* 13 */
  CNST_LIMB (0x179),  /* 14 */
  CNST_LIMB (0x262),  /* 15 */
  CNST_LIMB (0x3db),  /* 16 */
  CNST_LIMB (0x63d),  /* 17 */
  CNST_LIMB (0xa18),  /* 18 */
  CNST_LIMB (0x1055),  /* 19 */
  CNST_LIMB (0x1a6d),  /* 20 */
  CNST_LIMB (0x2ac2),  /* 21 */
  CNST_LIMB (0x452f),  /* 22 */
  CNST_LIMB (0x6ff1),  /* 23 */
  CNST_LIMB (0xb520),  /* 24 */
  CNST_LIMB (0x12511),  /* 25 */
  CNST_LIMB (0x1da31),  /* 26 */
  CNST_LIMB (0x2ff42),  /* 27 */
  CNST_LIMB (0x4d973),  /* 28 */
  CNST_LIMB (0x7d8b5),  /* 29 */
  CNST_LIMB (0xcb228),  /* 30 */
  CNST_LIMB (0x148add),  /* 31 */
  CNST_LIMB (0x213d05),  /* 32 */
  CNST_LIMB (0x35c7e2),  /* 33 */
  CNST_LIMB (0x5704e7),  /* 34 */
  CNST_LIMB (0x8cccc9),  /* 35 */
  CNST_LIMB (0xe3d1b0),  /* 36 */
  CNST_LIMB (0x1709e79),  /* 37 */
  CNST_LIMB (0x2547029),  /* 38 */
  CNST_LIMB (0x3c50ea2),  /* 39 */
  CNST_LIMB (0x6197ecb),  /* 40 */
  CNST_LIMB (0x9de8d6d),  /* 41 */
  CNST_LIMB (0xff80c38),  /* 42 */
  CNST_LIMB (0x19d699a5),  /* 43 */
  CNST_LIMB (0x29cea5dd),  /* 44 */
  CNST_LIMB (0x43a53f82),  /* 45 */
  CNST_LIMB (0x6d73e55f),  /* 46 */
  CNST_LIMB (0xb11924e1),  /* 47 */
  CNST_LIMB (0x11e8d0a40),  /* 48 */
  CNST_LIMB (0x1cfa62f21),  /* 49 */
  CNST_LIMB (0x2ee333961),  /* 50 */
  CNST_LIMB (0x4bdd96882),  /* 51 */
  CNST_LIMB (0x7ac0ca1e3),  /* 52 */
  CNST_LIMB (0xc69e60a65),  /* 53 */
  CNST_LIMB (0x1415f2ac48),  /* 54 */
  CNST_LIMB (0x207fd8b6ad),  /* 55 */
  CNST_LIMB (0x3495cb62f5),  /* 56 */
  CNST_LIMB (0x5515a419a2),  /* 57 */
  CNST_LIMB (0x89ab6f7c97),  /* 58 */
  CNST_LIMB (0xdec1139639),  /* 59 */
  CNST_LIMB (0x1686c8312d0),  /* 60 */
  CNST_LIMB (0x2472d96a909),  /* 61 */
  CNST_LIMB (0x3af9a19bbd9),  /* 62 */
  CNST_LIMB (0x5f6c7b064e2),  /* 63 */
  CNST_LIMB (0x9a661ca20bb),  /* 64 */
  CNST_LIMB (0xf9d297a859d),  /* 65 */
  CNST_LIMB (0x19438b44a658),  /* 66 */
  CNST_LIMB (0x28e0b4bf2bf5),  /* 67 */
  CNST_LIMB (0x42244003d24d),  /* 68 */
  CNST_LIMB (0x6b04f4c2fe42),  /* 69 */
  CNST_LIMB (0xad2934c6d08f),  /* 70 */
  CNST_LIMB (0x1182e2989ced1),  /* 71 */
  CNST_LIMB (0x1c5575e509f60),  /* 72 */
  CNST_LIMB (0x2dd8587da6e31),  /* 73 */
  CNST_LIMB (0x4a2dce62b0d91),  /* 74 */
  CNST_LIMB (0x780626e057bc2),  /* 75 */
  CNST_LIMB (0xc233f54308953),  /* 76 */
  CNST_LIMB (0x13a3a1c2360515),  /* 77 */
  CNST_LIMB (0x1fc6e116668e68),  /* 78 */
  CNST_LIMB (0x336a82d89c937d),  /* 79 */
  CNST_LIMB (0x533163ef0321e5),  /* 80 */
  CNST_LIMB (0x869be6c79fb562),  /* 81 */
  CNST_LIMB (0xd9cd4ab6a2d747),  /* 82 */
  CNST_LIMB (0x16069317e428ca9),  /* 83 */
  CNST_LIMB (0x23a367c34e563f0),  /* 84 */
  CNST_LIMB (0x39a9fadb327f099),  /* 85 */
  CNST_LIMB (0x5d4d629e80d5489),  /* 86 */
  CNST_LIMB (0x96f75d79b354522),  /* 87 */
  CNST_LIMB (0xf444c01834299ab),  /* 88 */
  CNST_LIMB (0x18b3c1d91e77decd),  /* 89 */
  CNST_LIMB (0x27f80ddaa1ba7878),  /* 90 */
  CNST_LIMB (0x40abcfb3c0325745),  /* 91 */
  CNST_LIMB (0x68a3dd8e61eccfbd),  /* 92 */
  CNST_LIMB (0xa94fad42221f2702),  /* 93 */
};
#define FIB_TABLE(n)  (__gmp_fib_table[(n)+1])

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

#define binvert_limb_table  __gmp_binvert_limb_table
const unsigned char  binvert_limb_table[128] = {
  0x01, 0xAB, 0xCD, 0xB7, 0x39, 0xA3, 0xC5, 0xEF,
  0xF1, 0x1B, 0x3D, 0xA7, 0x29, 0x13, 0x35, 0xDF,
  0xE1, 0x8B, 0xAD, 0x97, 0x19, 0x83, 0xA5, 0xCF,
  0xD1, 0xFB, 0x1D, 0x87, 0x09, 0xF3, 0x15, 0xBF,
  0xC1, 0x6B, 0x8D, 0x77, 0xF9, 0x63, 0x85, 0xAF,
  0xB1, 0xDB, 0xFD, 0x67, 0xE9, 0xD3, 0xF5, 0x9F,
  0xA1, 0x4B, 0x6D, 0x57, 0xD9, 0x43, 0x65, 0x8F,
  0x91, 0xBB, 0xDD, 0x47, 0xC9, 0xB3, 0xD5, 0x7F,
  0x81, 0x2B, 0x4D, 0x37, 0xB9, 0x23, 0x45, 0x6F,
  0x71, 0x9B, 0xBD, 0x27, 0xA9, 0x93, 0xB5, 0x5F,
  0x61, 0x0B, 0x2D, 0x17, 0x99, 0x03, 0x25, 0x4F,
  0x51, 0x7B, 0x9D, 0x07, 0x89, 0x73, 0x95, 0x3F,
  0x41, 0xEB, 0x0D, 0xF7, 0x79, 0xE3, 0x05, 0x2F,
  0x31, 0x5B, 0x7D, 0xE7, 0x69, 0x53, 0x75, 0x1F,
  0x21, 0xCB, 0xED, 0xD7, 0x59, 0xC3, 0xE5, 0x0F,
  0x11, 0x3B, 0x5D, 0xC7, 0x49, 0x33, 0x55, 0xFF
};

#define binvert_limb(inv,n)						\
  do {									\
    mp_limb_t  __n = (n);						\
    mp_limb_t  __inv;							\
    ASSERT ((__n & 1) == 1);						\
									\
    __inv = binvert_limb_table[(__n/2) & 0x7F]; /*  8 */		\
    if (GMP_NUMB_BITS > 8)   __inv = 2 * __inv - __inv * __inv * __n;	\
    if (GMP_NUMB_BITS > 16)  __inv = 2 * __inv - __inv * __inv * __n;	\
    if (GMP_NUMB_BITS > 32)  __inv = 2 * __inv - __inv * __inv * __n;	\
									\
    if (GMP_NUMB_BITS > 64)						\
      {									\
	int  __invbits = 64;						\
	do {								\
	  __inv = 2 * __inv - __inv * __inv * __n;			\
	  __invbits *= 2;						\
	} while (__invbits < GMP_NUMB_BITS);				\
      }									\
									\
    ASSERT ((__inv * __n & GMP_NUMB_MASK) == 1);			\
    (inv) = __inv & GMP_NUMB_MASK;					\
  } while (0)

#ifndef GMP_LIMB_BYTES
#define GMP_LIMB_BYTES sizeof(mp_limb_t)//SIZEOF_MP_LIMB_T
#endif

#define SIZEOF_UNSIGNED_LONG sizeof(unsigned long) //usually created by gmp's configure process via '#define SIZEOF_UNSIGNED_LONG $ac_cv_sizeof_unsigned_long' -- I think this is equivalent
#define BITS_PER_ULONG  (8 * SIZEOF_UNSIGNED_LONG)


/* mp_bases[10] data, as literal values */
#define MP_BASES_CHARS_PER_LIMB_10      19
#define MP_BASES_BIG_BASE_CTZ_10        19
#define MP_BASES_BIG_BASE_10            CNST_LIMB(0x8ac7230489e80000)
#define MP_BASES_BIG_BASE_INVERTED_10   CNST_LIMB(0xd83c94fb6d2ac34a)
#define MP_BASES_BIG_BASE_BINVERTED_10  CNST_LIMB(0x26b172506559ce15)
#define MP_BASES_NORMALIZATION_STEPS_10 0

/* Structure for conversion between internal binary format and strings.  */
struct bases
{
  /* Number of digits in the conversion base that always fits in an mp_limb_t.
     For example, for base 10 on a machine where an mp_limb_t has 32 bits this
     is 9, since 10**9 is the largest number that fits into an mp_limb_t.  */
  int chars_per_limb;

  /* log(2)/log(conversion_base) */
  mp_limb_t logb2;

  /* log(conversion_base)/log(2) */
  mp_limb_t log2b;

  /* base**chars_per_limb, i.e. the biggest number that fits a word, built by
     factors of base.  Exception: For 2, 4, 8, etc, big_base is log2(base),
     i.e. the number of bits used to represent each digit in the base.  */
  mp_limb_t big_base;

  /* A GMP_LIMB_BITS bit approximation to 1/big_base, represented as a
     fixed-point number.  Instead of dividing by big_base an application can
     choose to multiply by big_base_inverted.  */
  mp_limb_t big_base_inverted;
};


const struct bases mp_bases[257] =
{
  /*   0 */ { 0, 0, 0, 0, 0 },
  /*   1 */ { 0, 0, 0, 0, 0 },
  /*   2 */ { 64, CNST_LIMB(0xffffffffffffffff), CNST_LIMB(0x1fffffffffffffff), CNST_LIMB(0x1), CNST_LIMB(0x0) },
  /*   3 */ { 40, CNST_LIMB(0xa1849cc1a9a9e94e), CNST_LIMB(0x32b803473f7ad0f3), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d) },
  /*   4 */ { 32, CNST_LIMB(0x7fffffffffffffff), CNST_LIMB(0x3fffffffffffffff), CNST_LIMB(0x2), CNST_LIMB(0x0) },
  /*   5 */ { 27, CNST_LIMB(0x6e40d1a4143dcb94), CNST_LIMB(0x4a4d3c25e68dc57f), CNST_LIMB(0x6765c793fa10079d), CNST_LIMB(0x3ce9a36f23c0fc90) },
  /*   6 */ { 24, CNST_LIMB(0x6308c91b702a7cf4), CNST_LIMB(0x52b803473f7ad0f3), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295) },
  /*   7 */ { 22, CNST_LIMB(0x5b3064eb3aa6d388), CNST_LIMB(0x59d5d9fd5010b366), CNST_LIMB(0x3642798750226111), CNST_LIMB(0x2df495ccaa57147b) },
  /*   8 */ { 21, CNST_LIMB(0x5555555555555555), CNST_LIMB(0x5fffffffffffffff), CNST_LIMB(0x3), CNST_LIMB(0x0) },
  /*   9 */ { 20, CNST_LIMB(0x50c24e60d4d4f4a7), CNST_LIMB(0x6570068e7ef5a1e7), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d) },
  /*  10 */ { 19, CNST_LIMB(0x4d104d427de7fbcc), CNST_LIMB(0x6a4d3c25e68dc57f), CNST_LIMB(0x8ac7230489e80000), CNST_LIMB(0xd83c94fb6d2ac34a) },
  /*  11 */ { 18, CNST_LIMB(0x4a00270775914e88), CNST_LIMB(0x6eb3a9f01975077f), CNST_LIMB(0x4d28cb56c33fa539), CNST_LIMB(0xa8adf7ae45e7577b) },
  /*  12 */ { 17, CNST_LIMB(0x4768ce0d05818e12), CNST_LIMB(0x72b803473f7ad0f3), CNST_LIMB(0x1eca170c00000000), CNST_LIMB(0xa10c2bec5da8f8f) },
  /*  13 */ { 17, CNST_LIMB(0x452e53e365907bda), CNST_LIMB(0x766a008e4788cbcd), CNST_LIMB(0x780c7372621bd74d), CNST_LIMB(0x10f4becafe412ec3) },
  /*  14 */ { 16, CNST_LIMB(0x433cfffb4b5aae55), CNST_LIMB(0x79d5d9fd5010b366), CNST_LIMB(0x1e39a5057d810000), CNST_LIMB(0xf08480f672b4e86) },
  /*  15 */ { 16, CNST_LIMB(0x41867711b4f85355), CNST_LIMB(0x7d053f6d26089673), CNST_LIMB(0x5b27ac993df97701), CNST_LIMB(0x6779c7f90dc42f48) },
  /*  16 */ { 16, CNST_LIMB(0x3fffffffffffffff), CNST_LIMB(0x7fffffffffffffff), CNST_LIMB(0x4), CNST_LIMB(0x0) },
  /*  17 */ { 15, CNST_LIMB(0x3ea16afd58b10966), CNST_LIMB(0x82cc7edf592262cf), CNST_LIMB(0x27b95e997e21d9f1), CNST_LIMB(0x9c71e11bab279323) },
  /*  18 */ { 15, CNST_LIMB(0x3d64598d154dc4de), CNST_LIMB(0x8570068e7ef5a1e7), CNST_LIMB(0x5da0e1e53c5c8000), CNST_LIMB(0x5dfaa697ec6f6a1c) },
  /*  19 */ { 15, CNST_LIMB(0x3c43c23018bb5563), CNST_LIMB(0x87ef05ae409a0288), CNST_LIMB(0xd2ae3299c1c4aedb), CNST_LIMB(0x3711783f6be7e9ec) },
  /*  20 */ { 14, CNST_LIMB(0x3b3b9a42873069c7), CNST_LIMB(0x8a4d3c25e68dc57f), CNST_LIMB(0x16bcc41e90000000), CNST_LIMB(0x6849b86a12b9b01e) },
  /*  21 */ { 14, CNST_LIMB(0x3a4898f06cf41ac9), CNST_LIMB(0x8c8ddd448f8b845a), CNST_LIMB(0x2d04b7fdd9c0ef49), CNST_LIMB(0x6bf097ba5ca5e239) },
  /*  22 */ { 14, CNST_LIMB(0x39680b13582e7c18), CNST_LIMB(0x8eb3a9f01975077f), CNST_LIMB(0x5658597bcaa24000), CNST_LIMB(0x7b8015c8d7af8f08) },
  /*  23 */ { 14, CNST_LIMB(0x3897b2b751ae561a), CNST_LIMB(0x90c10500d63aa658), CNST_LIMB(0xa0e2073737609371), CNST_LIMB(0x975a24b3a3151b38) },
  /*  24 */ { 13, CNST_LIMB(0x37d5aed131f19c98), CNST_LIMB(0x92b803473f7ad0f3), CNST_LIMB(0xc29e98000000000), CNST_LIMB(0x50bd367972689db1) },
  /*  25 */ { 13, CNST_LIMB(0x372068d20a1ee5ca), CNST_LIMB(0x949a784bcd1b8afe), CNST_LIMB(0x14adf4b7320334b9), CNST_LIMB(0x8c240c4aecb13bb5) },
  /*  26 */ { 13, CNST_LIMB(0x3676867e5d60de29), CNST_LIMB(0x966a008e4788cbcd), CNST_LIMB(0x226ed36478bfa000), CNST_LIMB(0xdbd2e56854e118c9) },
  /*  27 */ { 13, CNST_LIMB(0x35d6deeb388df86f), CNST_LIMB(0x982809d5be7072db), CNST_LIMB(0x383d9170b85ff80b), CNST_LIMB(0x2351ffcaa9c7c4ae) },
  /*  28 */ { 13, CNST_LIMB(0x354071d61c77fa2e), CNST_LIMB(0x99d5d9fd5010b366), CNST_LIMB(0x5a3c23e39c000000), CNST_LIMB(0x6b24188ca33b0636) },
  /*  29 */ { 13, CNST_LIMB(0x34b260c5671b18ac), CNST_LIMB(0x9b74948f5532da4b), CNST_LIMB(0x8e65137388122bcd), CNST_LIMB(0xcc3dceaf2b8ba99d) },
  /*  30 */ { 13, CNST_LIMB(0x342be986572b45cc), CNST_LIMB(0x9d053f6d26089673), CNST_LIMB(0xdd41bb36d259e000), CNST_LIMB(0x2832e835c6c7d6b6) },
  /*  31 */ { 12, CNST_LIMB(0x33ac61b998fbbdf2), CNST_LIMB(0x9e88c6b3626a72aa), CNST_LIMB(0xaee5720ee830681), CNST_LIMB(0x76b6aa272e1873c5) },
  /*  32 */ { 12, CNST_LIMB(0x3333333333333333), CNST_LIMB(0x9fffffffffffffff), CNST_LIMB(0x5), CNST_LIMB(0x0) },
  /*  33 */ { 12, CNST_LIMB(0x32bfd90114c12861), CNST_LIMB(0xa16bad3758efd873), CNST_LIMB(0x172588ad4f5f0981), CNST_LIMB(0x61eaf5d402c7bf4f) },
  /*  34 */ { 12, CNST_LIMB(0x3251dcf6169e45f2), CNST_LIMB(0xa2cc7edf592262cf), CNST_LIMB(0x211e44f7d02c1000), CNST_LIMB(0xeeb658123ffb27ec) },
  /*  35 */ { 12, CNST_LIMB(0x31e8d59f180dc630), CNST_LIMB(0xa4231623369e78e5), CNST_LIMB(0x2ee56725f06e5c71), CNST_LIMB(0x5d5e3762e6fdf509) },
  /*  36 */ { 12, CNST_LIMB(0x3184648db8153e7a), CNST_LIMB(0xa570068e7ef5a1e7), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295) },
  /*  37 */ { 12, CNST_LIMB(0x312434e89c35dacd), CNST_LIMB(0xa6b3d78b6d3b24fb), CNST_LIMB(0x5b5b57f8a98a5dd1), CNST_LIMB(0x66ae7831762efb6f) },
  /*  38 */ { 12, CNST_LIMB(0x30c7fa349460a541), CNST_LIMB(0xa7ef05ae409a0288), CNST_LIMB(0x7dcff8986ea31000), CNST_LIMB(0x47388865a00f544) },
  /*  39 */ { 12, CNST_LIMB(0x306f6f4c8432bc6d), CNST_LIMB(0xa92203d587039cc1), CNST_LIMB(0xabd4211662a6b2a1), CNST_LIMB(0x7d673c33a123b54c) },
  /*  40 */ { 12, CNST_LIMB(0x301a557ffbfdd252), CNST_LIMB(0xaa4d3c25e68dc57f), CNST_LIMB(0xe8d4a51000000000), CNST_LIMB(0x19799812dea11197) },
  /*  41 */ { 11, CNST_LIMB(0x2fc873d1fda55f3b), CNST_LIMB(0xab7110e6ce866f2b), CNST_LIMB(0x7a32956ad081b79), CNST_LIMB(0xc27e62e0686feae) },
  /*  42 */ { 11, CNST_LIMB(0x2f799652a4e6dc49), CNST_LIMB(0xac8ddd448f8b845a), CNST_LIMB(0x9f49aaff0e86800), CNST_LIMB(0x9b6e7507064ce7c7) },
  /*  43 */ { 11, CNST_LIMB(0x2f2d8d8f64460aad), CNST_LIMB(0xada3f5fb9c415052), CNST_LIMB(0xce583bb812d37b3), CNST_LIMB(0x3d9ac2bf66cfed94) },
  /*  44 */ { 11, CNST_LIMB(0x2ee42e164e8f53a4), CNST_LIMB(0xaeb3a9f01975077f), CNST_LIMB(0x109b79a654c00000), CNST_LIMB(0xed46bc50ce59712a) },
  /*  45 */ { 11, CNST_LIMB(0x2e9d500984041dbd), CNST_LIMB(0xafbd42b465836767), CNST_LIMB(0x1543beff214c8b95), CNST_LIMB(0x813d97e2c89b8d46) },
  /*  46 */ { 11, CNST_LIMB(0x2e58cec05a6a8144), CNST_LIMB(0xb0c10500d63aa658), CNST_LIMB(0x1b149a79459a3800), CNST_LIMB(0x2e81751956af8083) },
  /*  47 */ { 11, CNST_LIMB(0x2e1688743ef9104c), CNST_LIMB(0xb1bf311e95d00de3), CNST_LIMB(0x224edfb5434a830f), CNST_LIMB(0xdd8e0a95e30c0988) },
  /*  48 */ { 11, CNST_LIMB(0x2dd65df7a583598f), CNST_LIMB(0xb2b803473f7ad0f3), CNST_LIMB(0x2b3fb00000000000), CNST_LIMB(0x7ad4dd48a0b5b167) },
  /*  49 */ { 11, CNST_LIMB(0x2d9832759d5369c4), CNST_LIMB(0xb3abb3faa02166cc), CNST_LIMB(0x3642798750226111), CNST_LIMB(0x2df495ccaa57147b) },
  /*  50 */ { 11, CNST_LIMB(0x2d5beb38dcd1394c), CNST_LIMB(0xb49a784bcd1b8afe), CNST_LIMB(0x43c33c1937564800), CNST_LIMB(0xe392010175ee5962) },
  /*  51 */ { 11, CNST_LIMB(0x2d216f7943e2ba6a), CNST_LIMB(0xb5848226989d33c3), CNST_LIMB(0x54411b2441c3cd8b), CNST_LIMB(0x84eaf11b2fe7738e) },
  /*  52 */ { 11, CNST_LIMB(0x2ce8a82efbb3ff2c), CNST_LIMB(0xb66a008e4788cbcd), CNST_LIMB(0x6851455acd400000), CNST_LIMB(0x3a1e3971e008995d) },
  /*  53 */ { 11, CNST_LIMB(0x2cb17fea7ad7e332), CNST_LIMB(0xb74b1fd64e0753c6), CNST_LIMB(0x80a23b117c8feb6d), CNST_LIMB(0xfd7a462344ffce25) },
  /*  54 */ { 11, CNST_LIMB(0x2c7be2b0cfa1ba50), CNST_LIMB(0xb82809d5be7072db), CNST_LIMB(0x9dff7d32d5dc1800), CNST_LIMB(0x9eca40b40ebcef8a) },
  /*  55 */ { 11, CNST_LIMB(0x2c47bddba92d7463), CNST_LIMB(0xb900e6160002ccfe), CNST_LIMB(0xc155af6faeffe6a7), CNST_LIMB(0x52fa161a4a48e43d) },
  /*  56 */ { 11, CNST_LIMB(0x2c14fffcaa8b131e), CNST_LIMB(0xb9d5d9fd5010b366), CNST_LIMB(0xebb7392e00000000), CNST_LIMB(0x1607a2cbacf930c1) },
  /*  57 */ { 10, CNST_LIMB(0x2be398c3a38be053), CNST_LIMB(0xbaa708f58014d37c), CNST_LIMB(0x50633659656d971), CNST_LIMB(0x97a014f8e3be55f1) },
  /*  58 */ { 10, CNST_LIMB(0x2bb378e758451068), CNST_LIMB(0xbb74948f5532da4b), CNST_LIMB(0x5fa8624c7fba400), CNST_LIMB(0x568df8b76cbf212c) },
  /*  59 */ { 10, CNST_LIMB(0x2b8492108be5e5f7), CNST_LIMB(0xbc3e9ca2e1a05533), CNST_LIMB(0x717d9faa73c5679), CNST_LIMB(0x20ba7c4b4e6ef492) },
  /*  60 */ { 10, CNST_LIMB(0x2b56d6c70d55481b), CNST_LIMB(0xbd053f6d26089673), CNST_LIMB(0x86430aac6100000), CNST_LIMB(0xe81ee46b9ef492f5) },
  /*  61 */ { 10, CNST_LIMB(0x2b2a3a608c72ddd5), CNST_LIMB(0xbdc899ab3ff56c5e), CNST_LIMB(0x9e64d9944b57f29), CNST_LIMB(0x9dc0d10d51940416) },
  /*  62 */ { 10, CNST_LIMB(0x2afeb0f1060c7e41), CNST_LIMB(0xbe88c6b3626a72aa), CNST_LIMB(0xba5ca5392cb0400), CNST_LIMB(0x5fa8ed2f450272a5) },
  /*  63 */ { 10, CNST_LIMB(0x2ad42f3c9aca595c), CNST_LIMB(0xbf45e08bcf06554e), CNST_LIMB(0xdab2ce1d022cd81), CNST_LIMB(0x2ba9eb8c5e04e641) },
  /*  64 */ { 10, CNST_LIMB(0x2aaaaaaaaaaaaaaa), CNST_LIMB(0xbfffffffffffffff), CNST_LIMB(0x6), CNST_LIMB(0x0) },
  /*  65 */ { 10, CNST_LIMB(0x2a82193a13425883), CNST_LIMB(0xc0b73cb42e16914c), CNST_LIMB(0x12aeed5fd3e2d281), CNST_LIMB(0xb67759cc00287bf1) },
  /*  66 */ { 10, CNST_LIMB(0x2a5a717672f66450), CNST_LIMB(0xc16bad3758efd873), CNST_LIMB(0x15c3da1572d50400), CNST_LIMB(0x78621feeb7f4ed33) },
  /*  67 */ { 10, CNST_LIMB(0x2a33aa6e56d9c71c), CNST_LIMB(0xc21d6713f453f356), CNST_LIMB(0x194c05534f75ee29), CNST_LIMB(0x43d55b5f72943bc0) },
  /*  68 */ { 10, CNST_LIMB(0x2a0dbbaa3bdfcea4), CNST_LIMB(0xc2cc7edf592262cf), CNST_LIMB(0x1d56299ada100000), CNST_LIMB(0x173decb64d1d4409) },
  /*  69 */ { 10, CNST_LIMB(0x29e89d244eb4bfaf), CNST_LIMB(0xc379084815b5774c), CNST_LIMB(0x21f2a089a4ff4f79), CNST_LIMB(0xe29fb54fd6b6074f) },
  /*  70 */ { 10, CNST_LIMB(0x29c44740d7db51e6), CNST_LIMB(0xc4231623369e78e5), CNST_LIMB(0x2733896c68d9a400), CNST_LIMB(0xa1f1f5c210d54e62) },
  /*  71 */ { 10, CNST_LIMB(0x29a0b2c743b14d74), CNST_LIMB(0xc4caba789e2b8687), CNST_LIMB(0x2d2cf2c33b533c71), CNST_LIMB(0x6aac7f9bfafd57b2) },
  /*  72 */ { 10, CNST_LIMB(0x297dd8dbb7c22a2d), CNST_LIMB(0xc570068e7ef5a1e7), CNST_LIMB(0x33f506e440000000), CNST_LIMB(0x3b563c2478b72ee2) },
  /*  73 */ { 10, CNST_LIMB(0x295bb2f9285c8c1b), CNST_LIMB(0xc6130af40bc0ecbf), CNST_LIMB(0x3ba43bec1d062211), CNST_LIMB(0x12b536b574e92d1b) },
  /*  74 */ { 10, CNST_LIMB(0x293a3aebe2be1c92), CNST_LIMB(0xc6b3d78b6d3b24fb), CNST_LIMB(0x4455872d8fd4e400), CNST_LIMB(0xdf86c03020404fa5) },
  /*  75 */ { 10, CNST_LIMB(0x29196acc815ebd9f), CNST_LIMB(0xc7527b930c965bf2), CNST_LIMB(0x4e2694539f2f6c59), CNST_LIMB(0xa34adf02234eea8e) },
  /*  76 */ { 10, CNST_LIMB(0x28f93cfb40f5c22a), CNST_LIMB(0xc7ef05ae409a0288), CNST_LIMB(0x5938006c18900000), CNST_LIMB(0x6f46eb8574eb59dd) },
  /*  77 */ { 10, CNST_LIMB(0x28d9ac1badc64117), CNST_LIMB(0xc88983ed6985bae5), CNST_LIMB(0x65ad9912474aa649), CNST_LIMB(0x42459b481df47cec) },
  /*  78 */ { 10, CNST_LIMB(0x28bab310a196b478), CNST_LIMB(0xc92203d587039cc1), CNST_LIMB(0x73ae9ff4241ec400), CNST_LIMB(0x1b424b95d80ca505) },
  /*  79 */ { 10, CNST_LIMB(0x289c4cf88b774469), CNST_LIMB(0xc9b892675266f66c), CNST_LIMB(0x836612ee9c4ce1e1), CNST_LIMB(0xf2c1b982203a0dac) },
  /*  80 */ { 10, CNST_LIMB(0x287e7529fb244e91), CNST_LIMB(0xca4d3c25e68dc57f), CNST_LIMB(0x9502f90000000000), CNST_LIMB(0xb7cdfd9d7bdbab7d) },
  /*  81 */ { 10, CNST_LIMB(0x286127306a6a7a53), CNST_LIMB(0xcae00d1cfdeb43cf), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d) },
  /*  82 */ { 10, CNST_LIMB(0x28445ec93f792b1e), CNST_LIMB(0xcb7110e6ce866f2b), CNST_LIMB(0xbebf59a07dab4400), CNST_LIMB(0x57931eeaf85cf64f) },
  /*  83 */ { 10, CNST_LIMB(0x282817e1038950fa), CNST_LIMB(0xcc0052b18b0e2a19), CNST_LIMB(0xd7540d4093bc3109), CNST_LIMB(0x305a944507c82f47) },
  /*  84 */ { 10, CNST_LIMB(0x280c4e90c9ab1f45), CNST_LIMB(0xcc8ddd448f8b845a), CNST_LIMB(0xf2b96616f1900000), CNST_LIMB(0xe007ccc9c22781a) },
  /*  85 */ { 9, CNST_LIMB(0x27f0ff1bc1ee87cd), CNST_LIMB(0xcd19bb053fb0284e), CNST_LIMB(0x336de62af2bca35), CNST_LIMB(0x3e92c42e000eeed4) },
  /*  86 */ { 9, CNST_LIMB(0x27d625ecf571c340), CNST_LIMB(0xcda3f5fb9c415052), CNST_LIMB(0x39235ec33d49600), CNST_LIMB(0x1ebe59130db2795e) },
  /*  87 */ { 9, CNST_LIMB(0x27bbbf95282fcd45), CNST_LIMB(0xce2c97d694adab3f), CNST_LIMB(0x3f674e539585a17), CNST_LIMB(0x268859e90f51b89) },
  /*  88 */ { 9, CNST_LIMB(0x27a1c8c8ddaf84da), CNST_LIMB(0xceb3a9f01975077f), CNST_LIMB(0x4645b6958000000), CNST_LIMB(0xd24cde0463108cfa) },
  /*  89 */ { 9, CNST_LIMB(0x27883e5e7df3f518), CNST_LIMB(0xcf393550f3aa6906), CNST_LIMB(0x4dcb74afbc49c19), CNST_LIMB(0xa536009f37adc383) },
  /*  90 */ { 9, CNST_LIMB(0x276f1d4c9847e90e), CNST_LIMB(0xcfbd42b465836767), CNST_LIMB(0x56064e1d18d9a00), CNST_LIMB(0x7cea06ce1c9ace10) },
  /*  91 */ { 9, CNST_LIMB(0x275662a841b30191), CNST_LIMB(0xd03fda8b97997f33), CNST_LIMB(0x5f04fe2cd8a39fb), CNST_LIMB(0x58db032e72e8ba43) },
  /*  92 */ { 9, CNST_LIMB(0x273e0ba38d15a47b), CNST_LIMB(0xd0c10500d63aa658), CNST_LIMB(0x68d74421f5c0000), CNST_LIMB(0x388cc17cae105447) },
  /*  93 */ { 9, CNST_LIMB(0x2726158c1b13cf03), CNST_LIMB(0xd140c9faa1e5439e), CNST_LIMB(0x738df1f6ab4827d), CNST_LIMB(0x1b92672857620ce0) },
  /*  94 */ { 9, CNST_LIMB(0x270e7dc9c01d8e9b), CNST_LIMB(0xd1bf311e95d00de3), CNST_LIMB(0x7f3afbc9cfb5e00), CNST_LIMB(0x18c6a9575c2ade4) },
  /*  95 */ { 9, CNST_LIMB(0x26f741dd3f070d61), CNST_LIMB(0xd23c41d42727c808), CNST_LIMB(0x8bf187fba88f35f), CNST_LIMB(0xd44da7da8e44b24f) },
  /*  96 */ { 9, CNST_LIMB(0x26e05f5f16c2159e), CNST_LIMB(0xd2b803473f7ad0f3), CNST_LIMB(0x99c600000000000), CNST_LIMB(0xaa2f78f1b4cc6794) },
  /*  97 */ { 9, CNST_LIMB(0x26c9d3fe61e80598), CNST_LIMB(0xd3327c6ab49ca6c8), CNST_LIMB(0xa8ce21eb6531361), CNST_LIMB(0x843c067d091ee4cc) },
  /*  98 */ { 9, CNST_LIMB(0x26b39d7fc6ddab08), CNST_LIMB(0xd3abb3faa02166cc), CNST_LIMB(0xb92112c1a0b6200), CNST_LIMB(0x62005e1e913356e3) },
  /*  99 */ { 9, CNST_LIMB(0x269db9bc7772a5cc), CNST_LIMB(0xd423b07e986aa967), CNST_LIMB(0xcad7718b8747c43), CNST_LIMB(0x4316eed01dedd518) },
  /* 100 */ { 9, CNST_LIMB(0x268826a13ef3fde6), CNST_LIMB(0xd49a784bcd1b8afe), CNST_LIMB(0xde0b6b3a7640000), CNST_LIMB(0x2725dd1d243aba0e) },
  /* 101 */ { 9, CNST_LIMB(0x2672e22d9dbdbd9f), CNST_LIMB(0xd510118708a8f8dd), CNST_LIMB(0xf2d8cf5fe6d74c5), CNST_LIMB(0xddd9057c24cb54f) },
  /* 102 */ { 9, CNST_LIMB(0x265dea72f169cc99), CNST_LIMB(0xd5848226989d33c3), CNST_LIMB(0x1095d25bfa712600), CNST_LIMB(0xedeee175a736d2a1) },
  /* 103 */ { 9, CNST_LIMB(0x26493d93a8cb2514), CNST_LIMB(0xd5f7cff41e09aeb8), CNST_LIMB(0x121b7c4c3698faa7), CNST_LIMB(0xc4699f3df8b6b328) },
  /* 104 */ { 9, CNST_LIMB(0x2634d9c282f3ef82), CNST_LIMB(0xd66a008e4788cbcd), CNST_LIMB(0x13c09e8d68000000), CNST_LIMB(0x9ebbe7d859cb5a7c) },
  /* 105 */ { 9, CNST_LIMB(0x2620bd41d8933adc), CNST_LIMB(0xd6db196a761949d9), CNST_LIMB(0x15876ccb0b709ca9), CNST_LIMB(0x7c828b9887eb2179) },
  /* 106 */ { 9, CNST_LIMB(0x260ce662ef04088a), CNST_LIMB(0xd74b1fd64e0753c6), CNST_LIMB(0x17723c2976da2a00), CNST_LIMB(0x5d652ab99001adcf) },
  /* 107 */ { 9, CNST_LIMB(0x25f95385547353fd), CNST_LIMB(0xd7ba18f93502e409), CNST_LIMB(0x198384e9c259048b), CNST_LIMB(0x4114f1754e5d7b32) },
  /* 108 */ { 9, CNST_LIMB(0x25e60316448db8e1), CNST_LIMB(0xd82809d5be7072db), CNST_LIMB(0x1bbde41dfeec0000), CNST_LIMB(0x274b7c902f7e0188) },
  /* 109 */ { 9, CNST_LIMB(0x25d2f390152f74f5), CNST_LIMB(0xd894f74b06ef8b40), CNST_LIMB(0x1e241d6e3337910d), CNST_LIMB(0xfc9e0fbb32e210c) },
  /* 110 */ { 9, CNST_LIMB(0x25c02379aa9ad043), CNST_LIMB(0xd900e6160002ccfe), CNST_LIMB(0x20b91cee9901ee00), CNST_LIMB(0xf4afa3e594f8ea1f) },
  /* 111 */ { 9, CNST_LIMB(0x25ad9165f2c18907), CNST_LIMB(0xd96bdad2acb5f5ef), CNST_LIMB(0x237ff9079863dfef), CNST_LIMB(0xcd85c32e9e4437b0) },
  /* 112 */ { 9, CNST_LIMB(0x259b3bf36735c90c), CNST_LIMB(0xd9d5d9fd5010b366), CNST_LIMB(0x267bf47000000000), CNST_LIMB(0xa9bbb147e0dd92a8) },
  /* 113 */ { 9, CNST_LIMB(0x258921cb955e7693), CNST_LIMB(0xda3ee7f38e181ed0), CNST_LIMB(0x29b08039fbeda7f1), CNST_LIMB(0x8900447b70e8eb82) },
  /* 114 */ { 9, CNST_LIMB(0x257741a2ac9170af), CNST_LIMB(0xdaa708f58014d37c), CNST_LIMB(0x2d213df34f65f200), CNST_LIMB(0x6b0a92adaad5848a) },
  /* 115 */ { 9, CNST_LIMB(0x25659a3711bc827d), CNST_LIMB(0xdb0e4126bcc86bd7), CNST_LIMB(0x30d201d957a7c2d3), CNST_LIMB(0x4f990ad8740f0ee5) },
  /* 116 */ { 9, CNST_LIMB(0x25542a50f84b9c39), CNST_LIMB(0xdb74948f5532da4b), CNST_LIMB(0x34c6d52160f40000), CNST_LIMB(0x3670a9663a8d3610) },
  /* 117 */ { 9, CNST_LIMB(0x2542f0c20000377d), CNST_LIMB(0xdbda071cc67e6db5), CNST_LIMB(0x3903f855d8f4c755), CNST_LIMB(0x1f5c44188057be3c) },
  /* 118 */ { 9, CNST_LIMB(0x2531ec64d772bd64), CNST_LIMB(0xdc3e9ca2e1a05533), CNST_LIMB(0x3d8de5c8ec59b600), CNST_LIMB(0xa2bea956c4e4977) },
  /* 119 */ { 9, CNST_LIMB(0x25211c1ce2fb5a6e), CNST_LIMB(0xdca258dca9331635), CNST_LIMB(0x4269541d1ff01337), CNST_LIMB(0xed68b23033c3637e) },
  /* 120 */ { 9, CNST_LIMB(0x25107ed5e7c3ec3b), CNST_LIMB(0xdd053f6d26089673), CNST_LIMB(0x479b38e478000000), CNST_LIMB(0xc99cf624e50549c5) },
  /* 121 */ { 9, CNST_LIMB(0x25001383bac8a744), CNST_LIMB(0xdd6753e032ea0efe), CNST_LIMB(0x4d28cb56c33fa539), CNST_LIMB(0xa8adf7ae45e7577b) },
  /* 122 */ { 9, CNST_LIMB(0x24efd921f390bce3), CNST_LIMB(0xddc899ab3ff56c5e), CNST_LIMB(0x5317871fa13aba00), CNST_LIMB(0x8a5bc740b1c113e5) },
  /* 123 */ { 9, CNST_LIMB(0x24dfceb3a26bb203), CNST_LIMB(0xde29142e0e01401f), CNST_LIMB(0x596d2f44de9fa71b), CNST_LIMB(0x6e6c7efb81cfbb9b) },
  /* 124 */ { 9, CNST_LIMB(0x24cff3430a0341a7), CNST_LIMB(0xde88c6b3626a72aa), CNST_LIMB(0x602fd125c47c0000), CNST_LIMB(0x54aba5c5cada5f10) },
  /* 125 */ { 9, CNST_LIMB(0x24c045e15c149931), CNST_LIMB(0xdee7b471b3a9507d), CNST_LIMB(0x6765c793fa10079d), CNST_LIMB(0x3ce9a36f23c0fc90) },
  /* 126 */ { 9, CNST_LIMB(0x24b0c5a679267ae2), CNST_LIMB(0xdf45e08bcf06554e), CNST_LIMB(0x6f15be069b847e00), CNST_LIMB(0x26fb43de2c8cd2a8) },
  /* 127 */ { 9, CNST_LIMB(0x24a171b0b31461c8), CNST_LIMB(0xdfa34e1177c23362), CNST_LIMB(0x7746b3e82a77047f), CNST_LIMB(0x12b94793db8486a1) },
  /* 128 */ { 9, CNST_LIMB(0x2492492492492492), CNST_LIMB(0xdfffffffffffffff), CNST_LIMB(0x7), CNST_LIMB(0x0) },
  /* 129 */ { 9, CNST_LIMB(0x24834b2c9d85cdfe), CNST_LIMB(0xe05bf942dbbc2145), CNST_LIMB(0x894953f7ea890481), CNST_LIMB(0xdd5deca404c0156d) },
  /* 130 */ { 9, CNST_LIMB(0x247476f924137501), CNST_LIMB(0xe0b73cb42e16914c), CNST_LIMB(0x932abffea4848200), CNST_LIMB(0xbd51373330291de0) },
  /* 131 */ { 9, CNST_LIMB(0x2465cbc00a40cec0), CNST_LIMB(0xe111cd1d5133412e), CNST_LIMB(0x9dacb687d3d6a163), CNST_LIMB(0x9fa4025d66f23085) },
  /* 132 */ { 9, CNST_LIMB(0x245748bc980e0427), CNST_LIMB(0xe16bad3758efd873), CNST_LIMB(0xa8d8102a44840000), CNST_LIMB(0x842530ee2db4949d) },
  /* 133 */ { 9, CNST_LIMB(0x2448ed2f49eb0633), CNST_LIMB(0xe1c4dfab90aab5ef), CNST_LIMB(0xb4b60f9d140541e5), CNST_LIMB(0x6aa7f2766b03dc25) },
  /* 134 */ { 9, CNST_LIMB(0x243ab85da36e3167), CNST_LIMB(0xe21d6713f453f356), CNST_LIMB(0xc15065d4856e4600), CNST_LIMB(0x53035ba7ebf32e8d) },
  /* 135 */ { 9, CNST_LIMB(0x242ca99203ea8c18), CNST_LIMB(0xe27545fba4fe385a), CNST_LIMB(0xceb1363f396d23c7), CNST_LIMB(0x3d12091fc9fb4914) },
  /* 136 */ { 9, CNST_LIMB(0x241ec01b7cce4ea0), CNST_LIMB(0xe2cc7edf592262cf), CNST_LIMB(0xdce31b2488000000), CNST_LIMB(0x28b1cb81b1ef1849) },
  /* 137 */ { 9, CNST_LIMB(0x2410fb4da9b3b0fc), CNST_LIMB(0xe323142dc8c66b55), CNST_LIMB(0xebf12a24bca135c9), CNST_LIMB(0x15c35be67ae3e2c9) },
  /* 138 */ { 9, CNST_LIMB(0x24035a808a0f315e), CNST_LIMB(0xe379084815b5774c), CNST_LIMB(0xfbe6f8dbf88f4a00), CNST_LIMB(0x42a17bd09be1ff0) },
  /* 139 */ { 8, CNST_LIMB(0x23f5dd105c67ab9d), CNST_LIMB(0xe3ce5d822ff4b643), CNST_LIMB(0x1ef156c084ce761), CNST_LIMB(0x8bf461f03cf0bbf) },
  /* 140 */ { 8, CNST_LIMB(0x23e8825d7b05abb1), CNST_LIMB(0xe4231623369e78e5), CNST_LIMB(0x20c4e3b94a10000), CNST_LIMB(0xf3fbb43f68a32d05) },
  /* 141 */ { 8, CNST_LIMB(0x23db49cc3a0866fe), CNST_LIMB(0xe4773465d54aded7), CNST_LIMB(0x22b0695a08ba421), CNST_LIMB(0xd84f44c48564dc19) },
  /* 142 */ { 8, CNST_LIMB(0x23ce32c4c6cfb9f5), CNST_LIMB(0xe4caba789e2b8687), CNST_LIMB(0x24b4f35d7a4c100), CNST_LIMB(0xbe58ebcce7956abe) },
  /* 143 */ { 8, CNST_LIMB(0x23c13cb308ab6ab7), CNST_LIMB(0xe51daa7e60fdd34c), CNST_LIMB(0x26d397284975781), CNST_LIMB(0xa5fac463c7c134b7) },
  /* 144 */ { 8, CNST_LIMB(0x23b4670682c0c709), CNST_LIMB(0xe570068e7ef5a1e7), CNST_LIMB(0x290d74100000000), CNST_LIMB(0x8f19241e28c7d757) },
  /* 145 */ { 8, CNST_LIMB(0x23a7b13237187c8b), CNST_LIMB(0xe5c1d0b53bc09fca), CNST_LIMB(0x2b63b3a37866081), CNST_LIMB(0x799a6d046c0ae1ae) },
  /* 146 */ { 8, CNST_LIMB(0x239b1aac8ac74728), CNST_LIMB(0xe6130af40bc0ecbf), CNST_LIMB(0x2dd789f4d894100), CNST_LIMB(0x6566e37d746a9e40) },
  /* 147 */ { 8, CNST_LIMB(0x238ea2ef2b24c379), CNST_LIMB(0xe663b741df9c37c0), CNST_LIMB(0x306a35e51b58721), CNST_LIMB(0x526887dbfb5f788f) },
  /* 148 */ { 8, CNST_LIMB(0x23824976f4045a26), CNST_LIMB(0xe6b3d78b6d3b24fb), CNST_LIMB(0x331d01712e10000), CNST_LIMB(0x408af3382b8efd3d) },
  /* 149 */ { 8, CNST_LIMB(0x23760dc3d6e4d729), CNST_LIMB(0xe7036db376537b90), CNST_LIMB(0x35f14200a827c61), CNST_LIMB(0x2fbb374806ec05f1) },
  /* 150 */ { 8, CNST_LIMB(0x2369ef58c30bd43e), CNST_LIMB(0xe7527b930c965bf2), CNST_LIMB(0x38e858b62216100), CNST_LIMB(0x1fe7c0f0afce87fe) },
  /* 151 */ { 8, CNST_LIMB(0x235dedbb8e82aa1c), CNST_LIMB(0xe7a102f9d39a9331), CNST_LIMB(0x3c03b2c13176a41), CNST_LIMB(0x11003d517540d32e) },
  /* 152 */ { 8, CNST_LIMB(0x23520874dfeb1ffd), CNST_LIMB(0xe7ef05ae409a0288), CNST_LIMB(0x3f44c9b21000000), CNST_LIMB(0x2f5810f98eff0dc) },
  /* 153 */ { 8, CNST_LIMB(0x23463f1019228dd7), CNST_LIMB(0xe83c856dd81804b7), CNST_LIMB(0x42ad23cef3113c1), CNST_LIMB(0xeb72e35e7840d910) },
  /* 154 */ { 8, CNST_LIMB(0x233a911b42aa9b3c), CNST_LIMB(0xe88983ed6985bae5), CNST_LIMB(0x463e546b19a2100), CNST_LIMB(0xd27de19593dc3614) },
  /* 155 */ { 8, CNST_LIMB(0x232efe26f7cf33f9), CNST_LIMB(0xe8d602d948f83829), CNST_LIMB(0x49f9fc3f96684e1), CNST_LIMB(0xbaf391fd3e5e6fc2) },
  /* 156 */ { 8, CNST_LIMB(0x232385c65381b485), CNST_LIMB(0xe92203d587039cc1), CNST_LIMB(0x4de1c9c5dc10000), CNST_LIMB(0xa4bd38c55228c81d) },
  /* 157 */ { 8, CNST_LIMB(0x2318278edde1b39b), CNST_LIMB(0xe96d887e26cd57b7), CNST_LIMB(0x51f77994116d2a1), CNST_LIMB(0x8fc5a8de8e1de782) },
  /* 158 */ { 8, CNST_LIMB(0x230ce3187a6c2be9), CNST_LIMB(0xe9b892675266f66c), CNST_LIMB(0x563cd6bb3398100), CNST_LIMB(0x7bf9265bea9d3a3b) },
  /* 159 */ { 8, CNST_LIMB(0x2301b7fd56ca21bb), CNST_LIMB(0xea03231d8d8224ba), CNST_LIMB(0x5ab3bb270beeb01), CNST_LIMB(0x69454b325983dccd) },
  /* 160 */ { 8, CNST_LIMB(0x22f6a5d9da38341c), CNST_LIMB(0xea4d3c25e68dc57f), CNST_LIMB(0x5f5e10000000000), CNST_LIMB(0x5798ee2308c39df9) },
  /* 161 */ { 8, CNST_LIMB(0x22ebac4c9580d89f), CNST_LIMB(0xea96defe264b59be), CNST_LIMB(0x643dce0ec16f501), CNST_LIMB(0x46e40ba0fa66a753) },
  /* 162 */ { 8, CNST_LIMB(0x22e0caf633834beb), CNST_LIMB(0xeae00d1cfdeb43cf), CNST_LIMB(0x6954fe21e3e8100), CNST_LIMB(0x3717b0870b0db3a7) },
  /* 163 */ { 8, CNST_LIMB(0x22d601796a418886), CNST_LIMB(0xeb28c7f233bdd372), CNST_LIMB(0x6ea5b9755f440a1), CNST_LIMB(0x2825e6775d11cdeb) },
  /* 164 */ { 8, CNST_LIMB(0x22cb4f7aec6fd8b4), CNST_LIMB(0xeb7110e6ce866f2b), CNST_LIMB(0x74322a1c0410000), CNST_LIMB(0x1a01a1c09d1b4dac) },
  /* 165 */ { 8, CNST_LIMB(0x22c0b4a15b80d83e), CNST_LIMB(0xebb8e95d3f7d9df2), CNST_LIMB(0x79fc8b6ae8a46e1), CNST_LIMB(0xc9eb0a8bebc8f3e) },
  /* 166 */ { 8, CNST_LIMB(0x22b630953a28f77a), CNST_LIMB(0xec0052b18b0e2a19), CNST_LIMB(0x80072a66d512100), CNST_LIMB(0xffe357ff59e6a004) },
  /* 167 */ { 8, CNST_LIMB(0x22abc300df54ca7c), CNST_LIMB(0xec474e39705912d2), CNST_LIMB(0x86546633b42b9c1), CNST_LIMB(0xe7dfd1be05fa61a8) },
  /* 168 */ { 8, CNST_LIMB(0x22a16b90698da5d2), CNST_LIMB(0xec8ddd448f8b845a), CNST_LIMB(0x8ce6b0861000000), CNST_LIMB(0xd11ed6fc78f760e5) },
  /* 169 */ { 8, CNST_LIMB(0x229729f1b2c83ded), CNST_LIMB(0xecd4011c8f11979a), CNST_LIMB(0x93c08e16a022441), CNST_LIMB(0xbb8db609dd29ebfe) },
  /* 170 */ { 8, CNST_LIMB(0x228cfdd444992f78), CNST_LIMB(0xed19bb053fb0284e), CNST_LIMB(0x9ae49717f026100), CNST_LIMB(0xa71aec8d1813d532) },
  /* 171 */ { 8, CNST_LIMB(0x2282e6e94ccb8588), CNST_LIMB(0xed5f0c3cbf8fa470), CNST_LIMB(0xa25577ae24c1a61), CNST_LIMB(0x93b612a9f20fbc02) },
  /* 172 */ { 8, CNST_LIMB(0x2278e4e392557ecf), CNST_LIMB(0xeda3f5fb9c415052), CNST_LIMB(0xaa15f068e610000), CNST_LIMB(0x814fc7b19a67d317) },
  /* 173 */ { 8, CNST_LIMB(0x226ef7776aa7fd29), CNST_LIMB(0xede87974f3c81855), CNST_LIMB(0xb228d6bf7577921), CNST_LIMB(0x6fd9a03f2e0a4b7c) },
  /* 174 */ { 8, CNST_LIMB(0x22651e5aaf5532d0), CNST_LIMB(0xee2c97d694adab3f), CNST_LIMB(0xba91158ef5c4100), CNST_LIMB(0x5f4615a38d0d316e) },
  /* 175 */ { 8, CNST_LIMB(0x225b5944b40b4694), CNST_LIMB(0xee7052491d2c3e64), CNST_LIMB(0xc351ad9aec0b681), CNST_LIMB(0x4f8876863479a286) },
  /* 176 */ { 8, CNST_LIMB(0x2251a7ee3cdfcca5), CNST_LIMB(0xeeb3a9f01975077f), CNST_LIMB(0xcc6db6100000000), CNST_LIMB(0x4094d8a3041b60eb) },
  /* 177 */ { 8, CNST_LIMB(0x22480a1174e913d9), CNST_LIMB(0xeef69fea211b2627), CNST_LIMB(0xd5e85d09025c181), CNST_LIMB(0x32600b8ed883a09b) },
  /* 178 */ { 8, CNST_LIMB(0x223e7f69e522683c), CNST_LIMB(0xef393550f3aa6906), CNST_LIMB(0xdfc4e816401c100), CNST_LIMB(0x24df8c6eb4b6d1f1) },
  /* 179 */ { 8, CNST_LIMB(0x223507b46b988abe), CNST_LIMB(0xef7b6b399471103e), CNST_LIMB(0xea06b4c72947221), CNST_LIMB(0x18097a8ee151acef) },
  /* 180 */ { 8, CNST_LIMB(0x222ba2af32dbbb9e), CNST_LIMB(0xefbd42b465836767), CNST_LIMB(0xf4b139365210000), CNST_LIMB(0xbd48cc8ec1cd8e3) },
  /* 181 */ { 8, CNST_LIMB(0x22225019a9b4d16c), CNST_LIMB(0xeffebccd41ffcd5c), CNST_LIMB(0xffc80497d520961), CNST_LIMB(0x3807a8d67485fb) },
  /* 182 */ { 8, CNST_LIMB(0x22190fb47b1af172), CNST_LIMB(0xf03fda8b97997f33), CNST_LIMB(0x10b4ebfca1dee100), CNST_LIMB(0xea5768860b62e8d8) },
  /* 183 */ { 8, CNST_LIMB(0x220fe14186679801), CNST_LIMB(0xf0809cf27f703d52), CNST_LIMB(0x117492de921fc141), CNST_LIMB(0xd54faf5b635c5005) },
  /* 184 */ { 8, CNST_LIMB(0x2206c483d7c6b786), CNST_LIMB(0xf0c10500d63aa658), CNST_LIMB(0x123bb2ce41000000), CNST_LIMB(0xc14a56233a377926) },
  /* 185 */ { 8, CNST_LIMB(0x21fdb93fa0e0ccc5), CNST_LIMB(0xf10113b153c8ea7b), CNST_LIMB(0x130a8b6157bdecc1), CNST_LIMB(0xae39a88db7cd329f) },
  /* 186 */ { 8, CNST_LIMB(0x21f4bf3a31bcdcaa), CNST_LIMB(0xf140c9faa1e5439e), CNST_LIMB(0x13e15dede0e8a100), CNST_LIMB(0x9c10bde69efa7ab6) },
  /* 187 */ { 8, CNST_LIMB(0x21ebd639f1d86584), CNST_LIMB(0xf18028cf72976a4e), CNST_LIMB(0x14c06d941c0ca7e1), CNST_LIMB(0x8ac36c42a2836497) },
  /* 188 */ { 8, CNST_LIMB(0x21e2fe06597361a6), CNST_LIMB(0xf1bf311e95d00de3), CNST_LIMB(0x15a7ff487a810000), CNST_LIMB(0x7a463c8b84f5ef67) },
  /* 189 */ { 8, CNST_LIMB(0x21da3667eb0e8ccb), CNST_LIMB(0xf1fde3d30e812642), CNST_LIMB(0x169859ddc5c697a1), CNST_LIMB(0x6a8e5f5ad090fd4b) },
  /* 190 */ { 8, CNST_LIMB(0x21d17f282d1a300e), CNST_LIMB(0xf23c41d42727c808), CNST_LIMB(0x1791c60f6fed0100), CNST_LIMB(0x5b91a2943596fc56) },
  /* 191 */ { 8, CNST_LIMB(0x21c8d811a3d3c9e1), CNST_LIMB(0xf27a4c0585cbf805), CNST_LIMB(0x18948e8c0e6fba01), CNST_LIMB(0x4d4667b1c468e8f0) },
  /* 192 */ { 8, CNST_LIMB(0x21c040efcb50f858), CNST_LIMB(0xf2b803473f7ad0f3), CNST_LIMB(0x19a1000000000000), CNST_LIMB(0x3fa39ab547994daf) },
  /* 193 */ { 8, CNST_LIMB(0x21b7b98f11b61c1a), CNST_LIMB(0xf2f56875eb3f2614), CNST_LIMB(0x1ab769203dafc601), CNST_LIMB(0x32a0a9b2faee1e2a) },
  /* 194 */ { 8, CNST_LIMB(0x21af41bcd19739ba), CNST_LIMB(0xf3327c6ab49ca6c8), CNST_LIMB(0x1bd81ab557f30100), CNST_LIMB(0x26357ceac0e96962) },
  /* 195 */ { 8, CNST_LIMB(0x21a6d9474c81adf0), CNST_LIMB(0xf36f3ffb6d916240), CNST_LIMB(0x1d0367a69fed1ba1), CNST_LIMB(0x1a5a6f65caa5859e) },
  /* 196 */ { 8, CNST_LIMB(0x219e7ffda5ad572a), CNST_LIMB(0xf3abb3faa02166cc), CNST_LIMB(0x1e39a5057d810000), CNST_LIMB(0xf08480f672b4e86) },
  /* 197 */ { 8, CNST_LIMB(0x219635afdcd3e46d), CNST_LIMB(0xf3e7d9379f70166a), CNST_LIMB(0x1f7b2a18f29ac3e1), CNST_LIMB(0x4383340615612ca) },
  /* 198 */ { 8, CNST_LIMB(0x218dfa2ec92d0643), CNST_LIMB(0xf423b07e986aa967), CNST_LIMB(0x20c850694c2aa100), CNST_LIMB(0xf3c77969ee4be5a2) },
  /* 199 */ { 8, CNST_LIMB(0x2185cd4c148e4ae2), CNST_LIMB(0xf45f3a98a20738a4), CNST_LIMB(0x222173cc014980c1), CNST_LIMB(0xe00993cc187c5ec9) },
  /* 200 */ { 8, CNST_LIMB(0x217daeda36ad7a5c), CNST_LIMB(0xf49a784bcd1b8afe), CNST_LIMB(0x2386f26fc1000000), CNST_LIMB(0xcd2b297d889bc2b6) },
  /* 201 */ { 8, CNST_LIMB(0x21759eac708452fe), CNST_LIMB(0xf4d56a5b33cec44a), CNST_LIMB(0x24f92ce8af296d41), CNST_LIMB(0xbb214d5064862b22) },
  /* 202 */ { 8, CNST_LIMB(0x216d9c96c7d490d4), CNST_LIMB(0xf510118708a8f8dd), CNST_LIMB(0x2678863cd0ece100), CNST_LIMB(0xa9e1a7ca7ea10e20) },
  /* 203 */ { 8, CNST_LIMB(0x2165a86e02cb358c), CNST_LIMB(0xf54a6e8ca5438db1), CNST_LIMB(0x280563f0a9472d61), CNST_LIMB(0x99626e72b39ea0cf) },
  /* 204 */ { 8, CNST_LIMB(0x215dc207a3c20fdf), CNST_LIMB(0xf5848226989d33c3), CNST_LIMB(0x29a02e1406210000), CNST_LIMB(0x899a5ba9c13fafd9) },
  /* 205 */ { 8, CNST_LIMB(0x2155e939e51e8b37), CNST_LIMB(0xf5be4d0cb51434aa), CNST_LIMB(0x2b494f4efe6d2e21), CNST_LIMB(0x7a80a705391e96ff) },
  /* 206 */ { 8, CNST_LIMB(0x214e1ddbb54cd933), CNST_LIMB(0xf5f7cff41e09aeb8), CNST_LIMB(0x2d0134ef21cbc100), CNST_LIMB(0x6c0cfe23de23042a) },
  /* 207 */ { 8, CNST_LIMB(0x21465fc4b2d68f98), CNST_LIMB(0xf6310b8f55304840), CNST_LIMB(0x2ec84ef4da2ef581), CNST_LIMB(0x5e377df359c944dd) },
  /* 208 */ { 8, CNST_LIMB(0x213eaecd2893dd60), CNST_LIMB(0xf66a008e4788cbcd), CNST_LIMB(0x309f102100000000), CNST_LIMB(0x50f8ac5fc8f53985) },
  /* 209 */ { 8, CNST_LIMB(0x21370ace09f681c6), CNST_LIMB(0xf6a2af9e5a0f0a08), CNST_LIMB(0x3285ee02a1420281), CNST_LIMB(0x44497266278e35b7) },
  /* 210 */ { 8, CNST_LIMB(0x212f73a0ef6db7cb), CNST_LIMB(0xf6db196a761949d9), CNST_LIMB(0x347d6104fc324100), CNST_LIMB(0x382316831f7ee175) },
  /* 211 */ { 8, CNST_LIMB(0x2127e92012e25004), CNST_LIMB(0xf7133e9b156c7be5), CNST_LIMB(0x3685e47dade53d21), CNST_LIMB(0x2c7f377833b8946e) },
  /* 212 */ { 8, CNST_LIMB(0x21206b264c4a39a7), CNST_LIMB(0xf74b1fd64e0753c6), CNST_LIMB(0x389ff6bb15610000), CNST_LIMB(0x2157c761ab4163ef) },
  /* 213 */ { 8, CNST_LIMB(0x2118f98f0e52c28f), CNST_LIMB(0xf782bdbfdda6577b), CNST_LIMB(0x3acc1912ebb57661), CNST_LIMB(0x16a7071803cc49a9) },
  /* 214 */ { 8, CNST_LIMB(0x211194366320dc66), CNST_LIMB(0xf7ba18f93502e409), CNST_LIMB(0x3d0acff111946100), CNST_LIMB(0xc6781d80f8224fc) },
  /* 215 */ { 8, CNST_LIMB(0x210a3af8e926bb78), CNST_LIMB(0xf7f1322182cf15d1), CNST_LIMB(0x3f5ca2e692eaf841), CNST_LIMB(0x294092d370a900b) },
  /* 216 */ { 8, CNST_LIMB(0x2102edb3d00e29a6), CNST_LIMB(0xf82809d5be7072db), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295) },
  /* 217 */ { 8, CNST_LIMB(0x20fbac44d5b6edc2), CNST_LIMB(0xf85ea0b0b27b2610), CNST_LIMB(0x443bcb714399a5c1), CNST_LIMB(0xe03b98f103fad6d2) },
  /* 218 */ { 8, CNST_LIMB(0x20f4768a4348ad08), CNST_LIMB(0xf894f74b06ef8b40), CNST_LIMB(0x46ca406c81af2100), CNST_LIMB(0xcee3d32cad2a9049) },
  /* 219 */ { 8, CNST_LIMB(0x20ed4c62ea57b1f0), CNST_LIMB(0xf8cb0e3b4b3bbdb3), CNST_LIMB(0x496e106ac22aaae1), CNST_LIMB(0xbe3f9df9277fdada) },
  /* 220 */ { 8, CNST_LIMB(0x20e62dae221c087a), CNST_LIMB(0xf900e6160002ccfe), CNST_LIMB(0x4c27d39fa5410000), CNST_LIMB(0xae46f0d94c05e933) },
  /* 221 */ { 8, CNST_LIMB(0x20df1a4bc4ba6525), CNST_LIMB(0xf9367f6da0ab2e9c), CNST_LIMB(0x4ef825c296e43ca1), CNST_LIMB(0x9ef2280fb437a33d) },
  /* 222 */ { 8, CNST_LIMB(0x20d8121c2c9e506e), CNST_LIMB(0xf96bdad2acb5f5ef), CNST_LIMB(0x51dfa61f5ad88100), CNST_LIMB(0x9039ff426d3f284b) },
  /* 223 */ { 8, CNST_LIMB(0x20d1150031e51549), CNST_LIMB(0xf9a0f8d3b0e04fde), CNST_LIMB(0x54def7a6d2f16901), CNST_LIMB(0x82178c6d6b51f8f4) },
  /* 224 */ { 8, CNST_LIMB(0x20ca22d927d8f54d), CNST_LIMB(0xf9d5d9fd5010b366), CNST_LIMB(0x57f6c10000000000), CNST_LIMB(0x74843b1ee4c1e053) },
  /* 225 */ { 8, CNST_LIMB(0x20c33b88da7c29aa), CNST_LIMB(0xfa0a7eda4c112ce6), CNST_LIMB(0x5b27ac993df97701), CNST_LIMB(0x6779c7f90dc42f48) },
  /* 226 */ { 8, CNST_LIMB(0x20bc5ef18c233bdf), CNST_LIMB(0xfa3ee7f38e181ed0), CNST_LIMB(0x5e7268b9bbdf8100), CNST_LIMB(0x5af23c74f9ad9fe9) },
  /* 227 */ { 8, CNST_LIMB(0x20b58cf5f31e4526), CNST_LIMB(0xfa7315d02f20c7bd), CNST_LIMB(0x61d7a7932ff3d6a1), CNST_LIMB(0x4ee7eae2acdc617e) },
  /* 228 */ { 8, CNST_LIMB(0x20aec5793770a74d), CNST_LIMB(0xfaa708f58014d37c), CNST_LIMB(0x65581f53c8c10000), CNST_LIMB(0x43556aa2ac262a0b) },
  /* 229 */ { 8, CNST_LIMB(0x20a8085ef096d530), CNST_LIMB(0xfadac1e711c832d1), CNST_LIMB(0x68f48a385b8320e1), CNST_LIMB(0x3835949593b8ddd1) },
  /* 230 */ { 8, CNST_LIMB(0x20a1558b2359c4b1), CNST_LIMB(0xfb0e4126bcc86bd7), CNST_LIMB(0x6cada69ed07c2100), CNST_LIMB(0x2d837fbe78458762) },
  /* 231 */ { 8, CNST_LIMB(0x209aace23fafa72e), CNST_LIMB(0xfb418734a9008bd9), CNST_LIMB(0x70843718cdbf27c1), CNST_LIMB(0x233a7e150a54a555) },
  /* 232 */ { 8, CNST_LIMB(0x20940e491ea988d7), CNST_LIMB(0xfb74948f5532da4b), CNST_LIMB(0x7479027ea1000000), CNST_LIMB(0x19561984a50ff8fe) },
  /* 233 */ { 8, CNST_LIMB(0x208d79a5006d7a47), CNST_LIMB(0xfba769b39e49640e), CNST_LIMB(0x788cd40268f39641), CNST_LIMB(0xfd211159fe3490f) },
  /* 234 */ { 8, CNST_LIMB(0x2086eedb8a3cead3), CNST_LIMB(0xfbda071cc67e6db5), CNST_LIMB(0x7cc07b437ecf6100), CNST_LIMB(0x6aa563e655033e3) },
  /* 235 */ { 8, CNST_LIMB(0x20806dd2c486dcc6), CNST_LIMB(0xfc0c6d447c5dd362), CNST_LIMB(0x8114cc6220762061), CNST_LIMB(0xfbb614b3f2d3b14c) },
  /* 236 */ { 8, CNST_LIMB(0x2079f67119059fae), CNST_LIMB(0xfc3e9ca2e1a05533), CNST_LIMB(0x858aa0135be10000), CNST_LIMB(0xeac0f8837fb05773) },
  /* 237 */ { 8, CNST_LIMB(0x2073889d50e7bf63), CNST_LIMB(0xfc7095ae91e1c760), CNST_LIMB(0x8a22d3b53c54c321), CNST_LIMB(0xda6e4c10e8615ca5) },
  /* 238 */ { 8, CNST_LIMB(0x206d243e9303d929), CNST_LIMB(0xfca258dca9331635), CNST_LIMB(0x8ede496339f34100), CNST_LIMB(0xcab755a8d01fa67f) },
  /* 239 */ { 8, CNST_LIMB(0x2066c93c62170aa8), CNST_LIMB(0xfcd3e6a0ca8906c2), CNST_LIMB(0x93bde80aec3a1481), CNST_LIMB(0xbb95a9ae71aa3e0c) },
  /* 240 */ { 8, CNST_LIMB(0x2060777e9b0db0f6), CNST_LIMB(0xfd053f6d26089673), CNST_LIMB(0x98c29b8100000000), CNST_LIMB(0xad0326c296b4f529) },
  /* 241 */ { 8, CNST_LIMB(0x205a2eed73563032), CNST_LIMB(0xfd3663b27f31d529), CNST_LIMB(0x9ded549671832381), CNST_LIMB(0x9ef9f21eed31b7c1) },
  /* 242 */ { 8, CNST_LIMB(0x2053ef71773d7e6a), CNST_LIMB(0xfd6753e032ea0efe), CNST_LIMB(0xa33f092e0b1ac100), CNST_LIMB(0x91747422be14b0b2) },
  /* 243 */ { 8, CNST_LIMB(0x204db8f388552ea9), CNST_LIMB(0xfd9810643d6614c3), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d) },
  /* 244 */ { 8, CNST_LIMB(0x20478b5cdbe2bb2f), CNST_LIMB(0xfdc899ab3ff56c5e), CNST_LIMB(0xae5b564ac3a10000), CNST_LIMB(0x77df79e9a96c06f6) },
  /* 245 */ { 8, CNST_LIMB(0x20416696f957cfbf), CNST_LIMB(0xfdf8f02086af2c4b), CNST_LIMB(0xb427f4b3be74c361), CNST_LIMB(0x6bc6019636c7d0c2) },
  /* 246 */ { 8, CNST_LIMB(0x203b4a8bb8d356e7), CNST_LIMB(0xfe29142e0e01401f), CNST_LIMB(0xba1f9a938041e100), CNST_LIMB(0x601c4205aebd9e47) },
  /* 247 */ { 8, CNST_LIMB(0x2035372541ab0f0d), CNST_LIMB(0xfe59063c8822ce56), CNST_LIMB(0xc0435871d1110f41), CNST_LIMB(0x54ddc59756f05016) },
  /* 248 */ { 8, CNST_LIMB(0x202f2c4e08fd6dcc), CNST_LIMB(0xfe88c6b3626a72aa), CNST_LIMB(0xc694446f01000000), CNST_LIMB(0x4a0648979c838c18) },
  /* 249 */ { 8, CNST_LIMB(0x202929f0d04b99e9), CNST_LIMB(0xfeb855f8ca88fb0d), CNST_LIMB(0xcd137a5b57ac3ec1), CNST_LIMB(0x3f91b6e0bb3a053d) },
  /* 250 */ { 8, CNST_LIMB(0x20232ff8a41b45eb), CNST_LIMB(0xfee7b471b3a9507d), CNST_LIMB(0xd3c21bcecceda100), CNST_LIMB(0x357c299a88ea76a5) },
  /* 251 */ { 8, CNST_LIMB(0x201d3e50daa036db), CNST_LIMB(0xff16e281db76303b), CNST_LIMB(0xdaa150410b788de1), CNST_LIMB(0x2bc1e517aecc56e3) },
  /* 252 */ { 8, CNST_LIMB(0x201754e5126d446d), CNST_LIMB(0xff45e08bcf06554e), CNST_LIMB(0xe1b24521be010000), CNST_LIMB(0x225f56ceb3da9f5d) },
  /* 253 */ { 8, CNST_LIMB(0x201173a1312ca135), CNST_LIMB(0xff74aef0efafadd7), CNST_LIMB(0xe8f62df12777c1a1), CNST_LIMB(0x1951136d53ad63ac) },
  /* 254 */ { 8, CNST_LIMB(0x200b9a71625f3b13), CNST_LIMB(0xffa34e1177c23362), CNST_LIMB(0xf06e445906fc0100), CNST_LIMB(0x1093d504b3cd7d93) },
  /* 255 */ { 8, CNST_LIMB(0x2005c94216230568), CNST_LIMB(0xffd1be4c7f2af942), CNST_LIMB(0xf81bc845c81bf801), CNST_LIMB(0x824794d1ec1814f) },
  /* 256 */ { 8, CNST_LIMB(0x1fffffffffffffff), CNST_LIMB(0xffffffffffffffff), CNST_LIMB(0x8), CNST_LIMB(0x0) },
};


typedef mp_limb_t UWtype;
typedef unsigned int UHWtype;
#define W_TYPE_SIZE GMP_LIMB_BITS

#ifndef FFT_TABLE_ATTRS
#define FFT_TABLE_ATTRS   static const
#endif

#define MPN_FFT_TABLE_SIZE  16


#define SIEVESIZE 512		/* FIXME: Allow gmp_init_primesieve to choose */
typedef struct
{
  unsigned long d;		   /* current index in s[] */
  unsigned long s0;		   /* number corresponding to s[0] */
  unsigned long sqrt_s0;	   /* misnomer for sqrt(s[SIEVESIZE-1]) */
  unsigned char s[SIEVESIZE + 1];  /* sieve table */
} gmp_primesieve_t;

ANYCALLER unsigned long int gmp_nextprime (gmp_primesieve_t *ps)
{
  unsigned long p, d, pi;
  unsigned char *sp;
  static unsigned char addtab[] =
    { 2,4,2,4,6,2,6,4,2,4,6,6,2,6,4,2,6,4,6,8,4,2,4,2,4,8,6,4,6,2,4,6,2,6,6,4,
      2,4,6,2,6,4,2,4,2,10,2,10 };
  unsigned char *addp = addtab;
  unsigned long ai;

  /* Look for already sieved primes.  A sentinel at the end of the sieving
     area allows us to use a very simple loop here.  */
  d = ps->d;
  sp = ps->s + d;
  while (*sp != 0)
    sp++;
  if (sp != ps->s + SIEVESIZE)
    {
      d = sp - ps->s;
      ps->d = d + 1;
      return ps->s0 + 2 * d;
    }

  /* Handle the number 2 separately.  */
  if (ps->s0 < 3)
    {
      ps->s0 = 3 - 2 * SIEVESIZE; /* Tricky */
      return 2;
    }

  /* Exhausted computed primes.  Resieve, then call ourselves recursively.  */

#if 0
  for (sp = ps->s; sp < ps->s + SIEVESIZE; sp++)
    *sp = 0;
#else
  memset (ps->s, 0, SIEVESIZE);
#endif

  ps->s0 += 2 * SIEVESIZE;

  /* Update sqrt_s0 as needed.  */
  while ((ps->sqrt_s0 + 1) * (ps->sqrt_s0 + 1) <= ps->s0 + 2 * SIEVESIZE - 1)
    ps->sqrt_s0++;

  pi = ((ps->s0 + 3) / 2) % 3;
  if (pi > 0)
    pi = 3 - pi;
  if (ps->s0 + 2 * pi <= 3)
    pi += 3;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 3;
    }

  pi = ((ps->s0 + 5) / 2) % 5;
  if (pi > 0)
    pi = 5 - pi;
  if (ps->s0 + 2 * pi <= 5)
    pi += 5;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 5;
    }

  pi = ((ps->s0 + 7) / 2) % 7;
  if (pi > 0)
    pi = 7 - pi;
  if (ps->s0 + 2 * pi <= 7)
    pi += 7;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 7;
    }

  p = 11;
  ai = 0;
  while (p <= ps->sqrt_s0)
    {
      pi = ((ps->s0 + p) / 2) % p;
      if (pi > 0)
	pi = p - pi;
      if (ps->s0 + 2 * pi <= p)
	  pi += p;
      sp = ps->s + pi;
      while (sp < ps->s + SIEVESIZE)
	{
	  *sp = 1, sp += p;
	}
      p += addp[ai];
      ai = (ai + 1) % 48;
    }
  ps->d = 0;
  return gmp_nextprime (ps);
}

/* Non-zero bit indicates a quadratic residue mod 0x100.
   This test identifies 82.81% as non-squares (212/256). */
static const mp_limb_t
sq_res_0x100[4] = {
  CNST_LIMB(0x202021202030213),
  CNST_LIMB(0x202021202020213),
  CNST_LIMB(0x202021202030212),
  CNST_LIMB(0x202021202020212),
};

/* 2^48-1 = 3^2 * 5 * 7 * 13 * 17 * 97 ... */
#define PERFSQR_MOD_BITS  49

/* bit count to limb count, rounding up */
#define BITS_TO_LIMBS(n)  (((n) + (GMP_NUMB_BITS - 1)) / GMP_NUMB_BITS)

#if defined (__GNUC__) && HAVE_HOST_CPU_FAMILY_x86
#if 0
/* FIXME: Check that these actually improve things.
   FIXME: Need a cld after each std.
   FIXME: Can't have inputs in clobbered registers, must describe them as
   dummy outputs, and add volatile. */
#define MPN_COPY_INCR(DST, SRC, N)					\
  __asm__ ("cld\n\trep\n\tmovsl" : :					\
	   "D" (DST), "S" (SRC), "c" (N) :				\
	   "cx", "di", "si", "memory")
#define MPN_COPY_DECR(DST, SRC, N)					\
  __asm__ ("std\n\trep\n\tmovsl" : :					\
	   "D" ((DST) + (N) - 1), "S" ((SRC) + (N) - 1), "c" (N) :	\
	   "cx", "di", "si", "memory")
#endif
#endif

#if defined (_CRAY)
#define MPN_COPY_DECR(dst, src, n)					\
  do {									\
    int __i;		/* Faster on some Crays with plain int */	\
    _Pragma ("_CRI ivdep");						\
    for (__i = (n) - 1; __i >= 0; __i--)				\
      (dst)[__i] = (src)[__i];						\
  } while (0)
#endif

#if ! defined (MPN_COPY_DECR) && HAVE_NATIVE_mpn_copyd
#define MPN_COPY_DECR(dst, src, size)					\
  do {									\
    ASSERT ((size) >= 0);						\
    ASSERT (MPN_SAME_OR_DECR_P (dst, src, size));			\
    mpn_copyd (dst, src, size);						\
  } while (0)
#endif

/* Copy N limbs from SRC to DST decrementing, N==0 allowed.  */
#if ! defined (MPN_COPY_DECR)
#define MPN_COPY_DECR(dst, src, n)					\
  do {									\
    ASSERT ((n) >= 0);							\
    ASSERT (MPN_SAME_OR_DECR_P (dst, src, n));				\
    if ((n) != 0)							\
      {									\
	mp_size_t __n = (n) - 1;					\
	mp_ptr __dst = (dst) + __n;					\
	mp_srcptr __src = (src) + __n;					\
	mp_limb_t __x;							\
	__x = *__src--;							\
	if (__n != 0)							\
	  {								\
	    do								\
	      {								\
		*__dst-- = __x;						\
		__x = *__src--;						\
	      }								\
	    while (--__n);						\
	  }								\
	*__dst-- = __x;							\
      }									\
  } while (0)
#endif

/* This test identifies 97.81% as non-squares. */
#define PERFSQR_MOD_TEST(up, usize) \
  do {                              \
    mp_limb_t  r;                   \
    PERFSQR_MOD_34 (r, up, usize);  \
                                    \
    /* 69.23% */                    \
    PERFSQR_MOD_2 (r, CNST_LIMB(91), CNST_LIMB(0xfd2fd2fd2fd3), \
                   CNST_LIMB(0x2191240), CNST_LIMB(0x8850a206953820e1)); \
                                    \
    /* 68.24% */                    \
    PERFSQR_MOD_2 (r, CNST_LIMB(85), CNST_LIMB(0xfcfcfcfcfcfd), \
                   CNST_LIMB(0x82158), CNST_LIMB(0x10b48c4b4206a105)); \
                                    \
    /* 55.56% */                    \
    PERFSQR_MOD_1 (r, CNST_LIMB( 9), CNST_LIMB(0xe38e38e38e39), \
                   CNST_LIMB(0x93)); \
                                    \
    /* 49.48% */                    \
    PERFSQR_MOD_2 (r, CNST_LIMB(97), CNST_LIMB(0xfd5c5f02a3a1), \
                   CNST_LIMB(0x1eb628b47), CNST_LIMB(0x6067981b8b451b5f)); \
  } while (0)

/* Grand total sq_res_0x100 and PERFSQR_MOD_TEST, 99.62% non-squares. */

/* helper for tests/mpz/t-perfsqr.c */
#define PERFSQR_DIVISORS  { 256, 91, 85, 9, 97, }

ANYCALLER void
gmp_init_primesieve (gmp_primesieve_t *ps)
{
  ps->s0 = 0;
  ps->sqrt_s0 = 0;
  ps->d = SIEVESIZE;
  ps->s[SIEVESIZE] = 0;		/* sentinel */
}



ANYCALLER static inline mp_size_t
mpn_mulmod_bnm1_itch (mp_size_t rn, mp_size_t an, mp_size_t bn) {
  mp_size_t n, itch;
  n = rn >> 1;
  itch = rn + 4 +
    (an > n ? (bn > n ? rn : n) : 0);
  return itch;
}


/* Compute the number of digits in base for nbits bits, making sure the result
   is never too small.  The two variants of the macro implement the same
   function; the GT2 variant below works just for bases > 2.  */
#define DIGITS_IN_BASE_FROM_BITS(res, nbits, b)				\
  do {									\
    mp_limb_t _ph, _dummy;						\
    size_t _nbits = (nbits);						\
    umul_ppmm (_ph, _dummy, mp_bases[b].logb2, _nbits);			\
    _ph += (_dummy + _nbits < _dummy);					\
    res = _ph + 1;							\
  } while (0)
#define DIGITS_IN_BASEGT2_FROM_BITS(res, nbits, b)			\
  do {									\
    mp_limb_t _ph, _dummy;						\
    size_t _nbits = (nbits);						\
    umul_ppmm (_ph, _dummy, mp_bases[b].logb2 + 1, _nbits);		\
    res = _ph + 1;							\
  } while (0)


#define MPN_SIZEINBASE(result, ptr, size, base)				\
  do {									\
    int	   __lb_base, __cnt;						\
    size_t __totbits;							\
									\
    ASSERT ((size) >= 0);						\
    ASSERT ((base) >= 2);						\
    ASSERT ((base) < numberof (mp_bases));				\
									\
    /* Special case for X == 0.  */					\
    if ((size) == 0)							\
      (result) = 1;							\
    else								\
      {									\
	/* Calculate the total number of significant bits of X.  */	\
	count_leading_zeros (__cnt, (ptr)[(size)-1]);			\
	__totbits = (size_t) (size) * GMP_NUMB_BITS - (__cnt - GMP_NAIL_BITS);\
									\
	if (POW2_P (base))						\
	  {								\
	    __lb_base = mp_bases[base].big_base;			\
	    (result) = (__totbits + __lb_base - 1) / __lb_base;		\
	  }								\
	else								\
	  {								\
	    DIGITS_IN_BASEGT2_FROM_BITS (result, __totbits, base);	\
	  }								\
      }									\
  } while (0)

#define MPN_SIZEINBASE_2EXP(result, ptr, size, base2exp)			\
  do {										\
    int          __cnt;								\
    mp_bitcnt_t  __totbits;							\
    ASSERT ((size) > 0);							\
    ASSERT ((ptr)[(size)-1] != 0);						\
    count_leading_zeros (__cnt, (ptr)[(size)-1]);				\
    __totbits = (mp_bitcnt_t) (size) * GMP_NUMB_BITS - (__cnt - GMP_NAIL_BITS);	\
    (result) = (__totbits + (base2exp)-1) / (base2exp);				\
  } while (0)

/* pre-inverse types for truncating division and modulo */
typedef struct {mp_limb_t inv32;} gmp_pi1_t;
typedef struct {mp_limb_t inv21, inv32, inv53;} gmp_pi2_t;

//Useful routines involving numbers/limbs, inlined completely - ported over from GMP
#if ! defined (MPN_COPY_INCR)
#define MPN_COPY_INCR(dst, src, n)					\
  do {									\
    ASSERT ((n) >= 0);							\
    ASSERT (MPN_SAME_OR_INCR_P (dst, src, n));				\
    if ((n) != 0)							\
      {									\
	mp_size_t __n = (n) - 1;					\
	mp_ptr __dst = (dst);						\
	mp_srcptr __src = (src);					\
	mp_limb_t __x;							\
	__x = *__src++;							\
	if (__n != 0)							\
	  {								\
	    do								\
	      {								\
		*__dst++ = __x;						\
		__x = *__src++;						\
	      }								\
	    while (--__n);						\
	  }								\
	*__dst++ = __x;							\
      }									\
  } while (0)
#endif

#ifndef MPN_COPY
#define MPN_COPY(d,s,n)							\
  do {									\
    ASSERT (MPN_SAME_OR_SEPARATE_P (d, s, n));				\
    MPN_COPY_INCR (d, s, n);						\
  } while (0)
#endif

#define ADDC_LIMB(cout, w, x, y)					\
  do {									\
    mp_limb_t  __x = (x);						\
    mp_limb_t  __y = (y);						\
    mp_limb_t  __w = __x + __y;						\
    (w) = __w;								\
    (cout) = __w < __x;							\
  } while (0)

#define SUBC_LIMB(cout, w, x, y)					\
  do {									\
    mp_limb_t  __x = (x);						\
    mp_limb_t  __y = (y);						\
    mp_limb_t  __w = __x - __y;						\
    (w) = __w;								\
    (cout) = __w > __x;							\
  } while (0)

#ifndef MPN_NORMALIZE
//TODO: This seems like it could cause warp divergence.
#define MPN_NORMALIZE(DST, NLIMBS) \
  do {									\
    while ((NLIMBS) > 0)						\
      {									\
	if ((DST)[(NLIMBS) - 1] != 0)					\
	  break;							\
	(NLIMBS)--;							\
      }									\
  } while (0)
#endif
#ifndef MPN_NORMALIZE_NOT_ZERO
//TODO: This seems like it could cause warp divergence.
#define MPN_NORMALIZE_NOT_ZERO(DST, NLIMBS)				\
  do {									\
    while (1)								\
      {									\
	ASSERT ((NLIMBS) >= 1);						\
	if ((DST)[(NLIMBS) - 1] != 0)					\
	  break;							\
	(NLIMBS)--;							\
      }									\
  } while (0)
#endif

/* For a threshold between algorithms A and B, size>=thresh is where B
   should be used.  Special value MP_SIZE_T_MAX means only ever use A, or
   value 0 means only ever use B.  The tests for these special values will
   be compile-time constants, so the compiler should be able to eliminate
   the code for the unwanted algorithm.  */

#if ! defined (__GNUC__) || __GNUC__ < 2
#define ABOVE_THRESHOLD(size,thresh)					\
  ((thresh) == 0							\
   || ((thresh) != MP_SIZE_T_MAX					\
       && (size) >= (thresh)))
#else
#define ABOVE_THRESHOLD(size,thresh)					\
  ((__builtin_constant_p (thresh) && (thresh) == 0)			\
   || (!(__builtin_constant_p (thresh) && (thresh) == MP_SIZE_T_MAX)	\
       && (size) >= (thresh)))
#endif
#define BELOW_THRESHOLD(size,thresh)  (! ABOVE_THRESHOLD (size, thresh))

#ifndef MPN_INCR_U
#if WANT_ASSERT
#define MPN_INCR_U(ptr, size, n)					\
  do {									\
    ASSERT ((size) >= 1);						\
    ASSERT_NOCARRY (mpn_add_1 (ptr, ptr, size, n));			\
  } while (0)
#else
#define MPN_INCR_U(ptr, size, n)   mpn_incr_u (ptr, n)
#endif
#endif

#ifndef MPN_DECR_U
#if WANT_ASSERT
#define MPN_DECR_U(ptr, size, n)					\
  do {									\
    ASSERT ((size) >= 1);						\
    ASSERT_NOCARRY (mpn_sub_1 (ptr, ptr, size, n));			\
  } while (0)
#else
#define MPN_DECR_U(ptr, size, n)   mpn_decr_u (ptr, n)
#endif
#endif

#define udiv_qrnnd_preinv(q, r, nh, nl, d, di)				\
  do {									\
    mp_limb_t _qh, _ql, _r, _mask;					\
    umul_ppmm (_qh, _ql, (nh), (di));					\
    if (__builtin_constant_p (nl) && (nl) == 0)				\
      {									\
	_qh += (nh) + 1;						\
	_r = - _qh * (d);						\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_qh += _mask;							\
	_r += _mask & (d);						\
      }									\
    else								\
      {									\
	add_ssaaaa (_qh, _ql, _qh, _ql, (nh) + 1, (nl));		\
	_r = (nl) - _qh * (d);						\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_qh += _mask;							\
	_r += _mask & (d);						\
	if (UNLIKELY (_r >= (d)))					\
	  {								\
	    _r -= (d);							\
	    _qh++;							\
	  }								\
      }									\
    (r) = _r;								\
    (q) = _qh;								\
  } while (0)


/* DIVEXACT_1_THRESHOLD is at what size to use mpn_divexact_1, as opposed to
   plain mpn_divrem_1.  Likewise BMOD_1_TO_MOD_1_THRESHOLD for
   mpn_modexact_1_odd against plain mpn_mod_1.  On most CPUs divexact and
   modexact are faster at all sizes, so the defaults are 0.  Those CPUs
   where this is not right have a tuned threshold.  */
#ifndef DIVEXACT_1_THRESHOLD
#define DIVEXACT_1_THRESHOLD  0
#endif
#ifndef BMOD_1_TO_MOD_1_THRESHOLD
#define BMOD_1_TO_MOD_1_THRESHOLD  10
#endif

#define MPN_MOD_OR_MODEXACT_1_ODD(src,size,divisor)			\
  (BELOW_THRESHOLD (size, BMOD_1_TO_MOD_1_THRESHOLD)			\
   ? mpn_modexact_1_odd (src, size, divisor)				\
   : mpn_mod_1 (src, size, divisor))

#if HAVE_NATIVE_mpn_modexact_1_odd
#define   mpn_modexact_1_odd  __MPN(modexact_1_odd)
__GMP_DECLSPEC mp_limb_t mpn_modexact_1_odd (mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;
#else
#define mpn_modexact_1_odd(src,size,divisor) \
  mpn_modexact_1c_odd (src, size, divisor, CNST_LIMB(0))
#endif

/* A bit mask of all the least significant zero bits of n, or -1 if n==0. */
#define LOW_ZEROS_MASK(n)  (((n) & -(n)) - 1)

typedef void gcd_subdiv_step_hook(void *, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, int);

/* Multiplicative inverse of 3, modulo 2^GMP_NUMB_BITS.
   Eg. 0xAAAAAAAB for 32 bits, 0xAAAAAAAAAAAAAAAB for 64 bits.
   GMP_NUMB_MAX/3*2+1 is right when GMP_NUMB_BITS is even, but when it's odd
   we need to start from GMP_NUMB_MAX>>1. */
#define MODLIMB_INVERSE_3 (((GMP_NUMB_MAX >> (GMP_NUMB_BITS % 2)) / 3) * 2 + 1)





/* (a/0), with a signed; is 1 if a=+/-1, 0 otherwise */
#define JACOBI_S0(a)   (((a) == 1) | ((a) == -1))

/* (a/0), with a unsigned; is 1 if a=+/-1, 0 otherwise */
#define JACOBI_U0(a)   ((a) == 1)

/* FIXME: JACOBI_LS0 and JACOBI_0LS are the same, so delete one and
   come up with a better name. */

/* (a/0), with a given by low and size;
   is 1 if a=+/-1, 0 otherwise */
#define JACOBI_LS0(alow,asize) \
  (((asize) == 1 || (asize) == -1) && (alow) == 1)

/* (a/0), with a an mpz_t;
   fetch of low limb always valid, even if size is zero */
#define JACOBI_Z0(a)   JACOBI_LS0 (PTR(a)[0], SIZ(a))

/* (0/b), with b unsigned; is 1 if b=1, 0 otherwise */
#define JACOBI_0U(b)   ((b) == 1)

/* (0/b), with b unsigned; is 1 if b=+/-1, 0 otherwise */
#define JACOBI_0S(b)   ((b) == 1 || (b) == -1)

/* (0/b), with b given by low and size; is 1 if b=+/-1, 0 otherwise */
#define JACOBI_0LS(blow,bsize) \
  (((bsize) == 1 || (bsize) == -1) && (blow) == 1)

/* Convert a bit1 to +1 or -1. */
#define JACOBI_BIT1_TO_PN(result_bit1) \
  (1 - ((int) (result_bit1) & 2))

/* (2/b), with b unsigned and odd;
   is (-1)^((b^2-1)/8) which is 1 if b==1,7mod8 or -1 if b==3,5mod8 and
   hence obtained from (b>>1)^b */
#define JACOBI_TWO_U_BIT1(b) \
  ((int) (((b) >> 1) ^ (b)))

/* (2/b)^twos, with b unsigned and odd */
#define JACOBI_TWOS_U_BIT1(twos, b) \
  ((int) ((twos) << 1) & JACOBI_TWO_U_BIT1 (b))

/* (2/b)^twos, with b unsigned and odd */
#define JACOBI_TWOS_U(twos, b) \
  (JACOBI_BIT1_TO_PN (JACOBI_TWOS_U_BIT1 (twos, b)))

/* (-1/b), with b odd (signed or unsigned);
   is (-1)^((b-1)/2) */
#define JACOBI_N1B_BIT1(b) \
  ((int) (b))

/* (a/b) effect due to sign of a: signed/unsigned, b odd;
   is (-1/b) if a<0, or +1 if a>=0 */
#define JACOBI_ASGN_SU_BIT1(a, b) \
  ((((a) < 0) << 1) & JACOBI_N1B_BIT1(b))

/* (a/b) effect due to sign of b: signed/signed;
   is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_SS_BIT1(a, b) \
  ((((a)<0) & ((b)<0)) << 1)

/* (a/b) effect due to sign of b: signed/mpz;
   is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_SZ_BIT1(a, b) \
  JACOBI_BSGN_SS_BIT1 (a, SIZ(b))

/* (a/b) effect due to sign of b: mpz/signed;
   is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_ZS_BIT1(a, b) \
  JACOBI_BSGN_SZ_BIT1 (b, a)

/* (a/b) reciprocity to switch to (b/a), a,b both unsigned and odd;
   is (-1)^((a-1)*(b-1)/4), which means +1 if either a,b==1mod4, or -1 if
   both a,b==3mod4, achieved in bit 1 by a&b.  No ASSERT()s about a,b odd
   because this is used in a couple of places with only bit 1 of a or b
   valid. */
#define JACOBI_RECIP_UU_BIT1(a, b) \
  ((int) ((a) & (b)))





#define MPN_DIVREM_OR_DIVEXACT_1(rp, up, n, d)				\
  do {									\
    if (BELOW_THRESHOLD (n, DIVEXACT_1_THRESHOLD))			\
      ASSERT_NOCARRY (mpn_divrem_1 (rp, (mp_size_t) 0, up, n, d));	\
    else								\
      {									\
	ASSERT (mpn_mod_1 (up, n, d) == 0);				\
	mpn_divexact_1 (rp, up, n, d);					\
      }									\
  } while (0)

/* Dividing (NH, NL) by D, returning the remainder only. Unlike
   udiv_qrnnd_preinv, works also for the case NH == D, where the
   quotient doesn't quite fit in a single limb. */
#define udiv_rnnd_preinv(r, nh, nl, d, di)				\
  do {									\
    mp_limb_t _qh, _ql, _r, _mask;					\
    umul_ppmm (_qh, _ql, (nh), (di));					\
    if (__builtin_constant_p (nl) && (nl) == 0)				\
      {									\
	_r = ~(_qh + (nh)) * (d);					\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_r += _mask & (d);						\
      }									\
    else								\
      {									\
	add_ssaaaa (_qh, _ql, _qh, _ql, (nh) + 1, (nl));		\
	_r = (nl) - _qh * (d);						\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_r += _mask & (d);						\
	if (UNLIKELY (_r >= (d)))					\
	  _r -= (d);							\
      }									\
    (r) = _r;								\
  } while (0)

#define invert_pi1(dinv, d1, d0)					\
  do {									\
    mp_limb_t _v, _p, _t1, _t0, _mask;					\
    invert_limb (_v, d1);						\
    _p = (d1) * _v;							\
    _p += (d0);								\
    if (_p < (d0))							\
      {									\
	_v--;								\
	_mask = -(mp_limb_t) (_p >= (d1));				\
	_p -= (d1);							\
	_v += _mask;							\
	_p -= _mask & (d1);						\
      }									\
    umul_ppmm (_t1, _t0, d0, _v);					\
    _p += _t1;								\
    if (_p < _t1)							\
      {									\
	_v--;								\
	if (UNLIKELY (_p >= (d1)))					\
	  {								\
	    if (_p > (d1) || _t0 >= (d0))				\
	      _v--;							\
	  }								\
      }									\
    (dinv).inv32 = _v;							\
  } while (0)

//assertation

#ifdef __LINE__
#define ASSERT_LINE  __LINE__
#else
#define ASSERT_LINE  -1
#endif

#ifdef __FILE__
#define ASSERT_FILE  __FILE__
#else
#define ASSERT_FILE  ""
#endif


void
__gmp_assert_header (const char *filename, int linenum)
{
  if (filename != NULL && filename[0] != '\0')
    {
      fprintf (stderr, "%s:", filename);
      if (linenum != -1)
        fprintf (stderr, "%d: ", linenum);
    }
}

struct hgcd_matrix
{
  mp_size_t alloc;		/* for sanity checking only */
  mp_size_t n;
  mp_ptr p[2][2];
};

/* The matrix non-negative M = (u, u'; v,v') keeps track of the
   reduction (a;b) = M (alpha; beta) where alpha, beta are smaller
   than a, b. The determinant must always be one, so that M has an
   inverse (v', -u'; -v, u). Elements always fit in GMP_NUMB_BITS - 1
   bits. */
struct hgcd_matrix1
{
  mp_limb_t u[2][2];
};

void
__gmp_assert_fail (const char *filename, int linenum,
                   const char *expr)
{
  __gmp_assert_header (filename, linenum);
  fprintf (stderr, "GNU MP assertion failed: %s\n", expr);
  abort();
}

#define MPN_HGCD_MATRIX_INIT_ITCH(n) (4 * ((n+1)/2 + 1))

typedef struct
{
  mp_limb_t d0, d1;
} mp_double_limb_t;

#define LIMB_HIGHBIT_TO_MASK(n)						\
  (((mp_limb_signed_t) -1 >> 1) < 0					\
   ? (mp_limb_signed_t) (n) >> (GMP_LIMB_BITS - 1)			\
   : (n) & GMP_LIMB_HIGHBIT ? MP_LIMB_T_MAX : CNST_LIMB(0))

#define MP_LIMB_T_SWAP(x, y)						\
  do {									\
    mp_limb_t __mp_limb_t_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_limb_t_swap__tmp;					\
  } while (0)
#define MP_SIZE_T_SWAP(x, y)						\
  do {									\
    mp_size_t __mp_size_t_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_size_t_swap__tmp;					\
  } while (0)

#define MP_PTR_SWAP(x, y)						\
  do {									\
    mp_ptr __mp_ptr_swap__tmp = (x);					\
    (x) = (y);								\
    (y) = __mp_ptr_swap__tmp;						\
  } while (0)
#define MP_SRCPTR_SWAP(x, y)						\
  do {									\
    mp_srcptr __mp_srcptr_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_srcptr_swap__tmp;					\
  } while (0)

#define MPN_PTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    MP_PTR_SWAP (xp, yp);						\
    MP_SIZE_T_SWAP (xs, ys);						\
  } while(0)
#define MPN_SRCPTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    MP_SRCPTR_SWAP (xp, yp);						\
    MP_SIZE_T_SWAP (xs, ys);						\
  } while(0)

#define MPZ_PTR_SWAP(x, y)						\
  do {									\
    mpz_ptr __mpz_ptr_swap__tmp = (x);					\
    (x) = (y);								\
    (y) = __mpz_ptr_swap__tmp;						\
  } while (0)
#define MPZ_SRCPTR_SWAP(x, y)						\
  do {									\
    mpz_srcptr __mpz_srcptr_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mpz_srcptr_swap__tmp;					\
  } while (0)

#define MPQ_PTR_SWAP(x, y)						\
  do {                                                                  \
    mpq_ptr __mpq_ptr_swap__tmp = (x);					\
    (x) = (y);                                                          \
    (y) = __mpq_ptr_swap__tmp;						\
  } while (0)
#define MPQ_SRCPTR_SWAP(x, y)                                           \
  do {                                                                  \
    mpq_srcptr __mpq_srcptr_swap__tmp = (x);                            \
    (x) = (y);                                                          \
    (y) = __mpq_srcptr_swap__tmp;                                       \
  } while (0)

/* Assert that an mpn region {ptr,size} is zero, or non-zero.
   size==0 is allowed, and in that case {ptr,size} considered to be zero.  */
#if WANT_ASSERT
#define ASSERT_MPN_ZERO_P(ptr,size)					\
  do {									\
    mp_size_t  __i;							\
    ASSERT ((size) >= 0);						\
    for (__i = 0; __i < (size); __i++)					\
      ASSERT ((ptr)[__i] == 0);						\
  } while (0)
#define ASSERT_MPN_NONZERO_P(ptr,size)					\
  do {									\
    mp_size_t  __i;							\
    int	       __nonzero = 0;						\
    ASSERT ((size) >= 0);						\
    for (__i = 0; __i < (size); __i++)					\
      if ((ptr)[__i] != 0)						\
	{								\
	  __nonzero = 1;						\
	  break;							\
	}								\
    ASSERT (__nonzero);							\
  } while (0)
#else
#define ASSERT_MPN_ZERO_P(ptr,size)     do {} while (0)
#define ASSERT_MPN_NONZERO_P(ptr,size)  do {} while (0)
#endif

#define ASSERT_FAIL(expr)  __gmp_assert_fail (ASSERT_FILE, ASSERT_LINE, #expr)

#define ASSERT_ALWAYS(expr)						\
  do {									\
    if (UNLIKELY (!(expr)))						\
      ASSERT_FAIL (expr);						\
  } while (0)

/* ASSERT_CARRY checks the expression is non-zero, and ASSERT_NOCARRY checks
   that it's zero.  In both cases if assertion checking is disabled the
   expression is still evaluated.  These macros are meant for use with
   routines like mpn_add_n() where the return value represents a carry or
   whatever that should or shouldn't occur in some context.  For example,
   ASSERT_NOCARRY (mpn_add_n (rp, s1p, s2p, size)); */
#if WANT_ASSERT
#define ASSERT_CARRY(expr)     ASSERT_ALWAYS ((expr) != 0)
#define ASSERT_NOCARRY(expr)   ASSERT_ALWAYS ((expr) == 0)
#else
#define ASSERT_CARRY(expr)     (expr)
#define ASSERT_NOCARRY(expr)   (expr)
#endif

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

struct gcdext_ctx
{
  /* Result parameters. */
  mp_ptr gp;
  mp_size_t gn;
  mp_ptr up;
  mp_size_t *usize;

  /* Cofactors updated in each step. */
  mp_size_t un;
  mp_ptr u0, u1, tp;
};

/* Extract one numb, shifting count bits left
    ________  ________
   |___xh___||___xl___|
	  |____r____|
   >count <

   The count includes any nail bits, so it should work fine if count
   is computed using count_leading_zeros. If GMP_NAIL_BITS > 0, all of
   xh, xl and r include nail bits. Must have 0 < count < GMP_LIMB_BITS.

   FIXME: Omit masking with GMP_NUMB_MASK, and let callers do that for
   those calls where the count high bits of xh may be non-zero.
*/

/* 4*(an + 1) + 4*(bn + 1) + an */
#define MPN_GCDEXT_LEHMER_ITCH(an, bn) (5*(an) + 4*(bn) + 8)

#ifndef HGCD_THRESHOLD
#define HGCD_THRESHOLD 400
#endif

#ifndef HGCD_APPR_THRESHOLD
#define HGCD_APPR_THRESHOLD 400
#endif

#ifndef HGCD_REDUCE_THRESHOLD
#define HGCD_REDUCE_THRESHOLD 1000
#endif

#ifndef GCD_DC_THRESHOLD
#define GCD_DC_THRESHOLD 1000
#endif

#ifndef GCDEXT_DC_THRESHOLD
#define GCDEXT_DC_THRESHOLD 600
#endif

#define MPN_GCDEXT_LEHMER_N_ITCH(n) (4*(n) + 3)

#define MPN_EXTRACT_NUMB(count, xh, xl)				\
  ((((xh) << ((count) - GMP_NAIL_BITS)) & GMP_NUMB_MASK) |	\
   ((xl) >> (GMP_LIMB_BITS - (count))))

/* Needs storage for the quotient */
#define MPN_GCD_SUBDIV_STEP_ITCH(n) (n)

/* Check that the nail parts are zero. */
#define ASSERT_ALWAYS_LIMB(limb)					\
  do {									\
    mp_limb_t  __nail = (limb) & GMP_NAIL_MASK;				\
    ASSERT_ALWAYS (__nail == 0);					\
  } while (0)
#define ASSERT_ALWAYS_MPN(ptr, size)					\
  do {									\
    /* let whole loop go dead when no nails */				\
    if (GMP_NAIL_BITS != 0)						\
      {									\
	mp_size_t  __i;							\
	for (__i = 0; __i < (size); __i++)				\
	  ASSERT_ALWAYS_LIMB ((ptr)[__i]);				\
      }									\
  } while (0)

#if WANT_ASSERT
#define ASSERT_LIMB(limb)       ASSERT_ALWAYS_LIMB (limb)
#define ASSERT_MPN(ptr, size)   ASSERT_ALWAYS_MPN (ptr, size)
#else
#define ASSERT_LIMB(limb)       do {} while (0)
#define ASSERT_MPN(ptr, size)   do {} while (0)
#endif

/* Definitions for mpn_set_str and mpn_get_str */
struct powers
{
  mp_ptr p;			/* actual power value */
  mp_size_t n;			/* # of limbs at p */
  mp_size_t shift;		/* weight of lowest limb, in limb base B */
  size_t digits_in_base;	/* number of corresponding digits */
  int base;
};
typedef struct powers powers_t;
#define mpn_str_powtab_alloc(n) ((n) + 2 * GMP_LIMB_BITS) /* FIXME: This can perhaps be trimmed */
#define mpn_dc_set_str_itch(n) ((n) + GMP_LIMB_BITS)
#define mpn_dc_get_str_itch(n) ((n) + GMP_LIMB_BITS)

#define MPZ_REALLOC(z,n) (UNLIKELY ((n) > ALLOC(z))			\
			  ? (mp_ptr) _mpz_realloc(z,n)			\
			  : PTR(z))

#define numberof(x)  (sizeof (x) / sizeof ((x)[0]))

// /- /- /- /- /- /- /- Craploads of gmp-impl.h internal constants...

#ifndef SET_STR_DC_THRESHOLD
#define SET_STR_DC_THRESHOLD            750
#endif

#ifndef MUL_TOOM22_THRESHOLD
#define MUL_TOOM22_THRESHOLD             30
#endif

#ifndef MUL_TOOM33_THRESHOLD
#define MUL_TOOM33_THRESHOLD            100
#endif

#ifndef MUL_TOOM44_THRESHOLD
#define MUL_TOOM44_THRESHOLD            300
#endif

#ifndef MUL_TOOM6H_THRESHOLD
#define MUL_TOOM6H_THRESHOLD            350
#endif

#ifndef SQR_TOOM6_THRESHOLD
#define SQR_TOOM6_THRESHOLD MUL_TOOM6H_THRESHOLD
#endif

#ifndef MUL_TOOM8H_THRESHOLD
#define MUL_TOOM8H_THRESHOLD            450
#endif

#ifndef SQR_TOOM8_THRESHOLD
#define SQR_TOOM8_THRESHOLD MUL_TOOM8H_THRESHOLD
#endif

#ifndef MUL_TOOM32_TO_TOOM43_THRESHOLD
#define MUL_TOOM32_TO_TOOM43_THRESHOLD  100
#endif

#ifndef MUL_TOOM32_TO_TOOM53_THRESHOLD
#define MUL_TOOM32_TO_TOOM53_THRESHOLD  110
#endif

#ifndef MUL_TOOM42_TO_TOOM53_THRESHOLD
#define MUL_TOOM42_TO_TOOM53_THRESHOLD  100
#endif

#ifndef MUL_TOOM42_TO_TOOM63_THRESHOLD
#define MUL_TOOM42_TO_TOOM63_THRESHOLD  110
#endif

#ifndef MUL_TOOM43_TO_TOOM54_THRESHOLD
#define MUL_TOOM43_TO_TOOM54_THRESHOLD  150
#endif

/* MUL_TOOM22_THRESHOLD_LIMIT is the maximum for MUL_TOOM22_THRESHOLD.  In a
   normal build MUL_TOOM22_THRESHOLD is a constant and we use that.  In a fat
   binary or tune program build MUL_TOOM22_THRESHOLD is a variable and a
   separate hard limit will have been defined.  Similarly for TOOM3.  */
#ifndef MUL_TOOM22_THRESHOLD_LIMIT
#define MUL_TOOM22_THRESHOLD_LIMIT  MUL_TOOM22_THRESHOLD
#endif
#ifndef MUL_TOOM33_THRESHOLD_LIMIT
#define MUL_TOOM33_THRESHOLD_LIMIT  MUL_TOOM33_THRESHOLD
#endif
#ifndef MULLO_BASECASE_THRESHOLD_LIMIT
#define MULLO_BASECASE_THRESHOLD_LIMIT  MULLO_BASECASE_THRESHOLD
#endif
#ifndef SQRLO_BASECASE_THRESHOLD_LIMIT
#define SQRLO_BASECASE_THRESHOLD_LIMIT  SQRLO_BASECASE_THRESHOLD
#endif
#ifndef SQRLO_DC_THRESHOLD_LIMIT
#define SQRLO_DC_THRESHOLD_LIMIT  SQRLO_DC_THRESHOLD
#endif

/* SQR_BASECASE_THRESHOLD is where mpn_sqr_basecase should take over from
   mpn_mul_basecase.  Default is to use mpn_sqr_basecase from 0.  (Note that we
   certainly always want it if there's a native assembler mpn_sqr_basecase.)

   If it turns out that mpn_toom2_sqr becomes faster than mpn_mul_basecase
   before mpn_sqr_basecase does, then SQR_BASECASE_THRESHOLD is the toom2
   threshold and SQR_TOOM2_THRESHOLD is 0.  This oddity arises more or less
   because SQR_TOOM2_THRESHOLD represents the size up to which mpn_sqr_basecase
   should be used, and that may be never.  */

#ifndef SQR_BASECASE_THRESHOLD
#define SQR_BASECASE_THRESHOLD            0  /* never use mpn_mul_basecase */
#endif

#ifndef SQR_TOOM2_THRESHOLD
#define SQR_TOOM2_THRESHOLD              50
#endif

#ifndef SQR_TOOM3_THRESHOLD
#define SQR_TOOM3_THRESHOLD             120
#endif

#ifndef SQR_TOOM4_THRESHOLD
#define SQR_TOOM4_THRESHOLD             400
#endif

/* See comments above about MUL_TOOM33_THRESHOLD_LIMIT.  */
#ifndef SQR_TOOM3_THRESHOLD_LIMIT
#define SQR_TOOM3_THRESHOLD_LIMIT  SQR_TOOM3_THRESHOLD
#endif

#ifndef MULMID_TOOM42_THRESHOLD
#define MULMID_TOOM42_THRESHOLD     MUL_TOOM22_THRESHOLD
#endif

#ifndef MULLO_BASECASE_THRESHOLD
#define MULLO_BASECASE_THRESHOLD          0  /* never use mpn_mul_basecase */
#endif

#ifndef MULLO_DC_THRESHOLD
#define MULLO_DC_THRESHOLD         (2*MUL_TOOM22_THRESHOLD)
#endif

#ifndef MULLO_MUL_N_THRESHOLD
#define MULLO_MUL_N_THRESHOLD      (2*MUL_FFT_THRESHOLD)
#endif

#ifndef SQRLO_BASECASE_THRESHOLD
#define SQRLO_BASECASE_THRESHOLD          0  /* never use mpn_sqr_basecase */
#endif

#ifndef SQRLO_DC_THRESHOLD
#define SQRLO_DC_THRESHOLD         (MULLO_DC_THRESHOLD)
#endif

#ifndef SQRLO_SQR_THRESHOLD
#define SQRLO_SQR_THRESHOLD        (MULLO_MUL_N_THRESHOLD)
#endif

#ifndef DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD        (2*MUL_TOOM22_THRESHOLD)
#endif

#ifndef DC_DIVAPPR_Q_THRESHOLD
#define DC_DIVAPPR_Q_THRESHOLD          200
#endif

#ifndef DC_BDIV_QR_THRESHOLD
#define DC_BDIV_QR_THRESHOLD       (2*MUL_TOOM22_THRESHOLD)
#endif

#ifndef DC_BDIV_Q_THRESHOLD
#define DC_BDIV_Q_THRESHOLD             180
#endif

#ifndef DIVEXACT_JEB_THRESHOLD
#define DIVEXACT_JEB_THRESHOLD           25
#endif

#ifndef INV_MULMOD_BNM1_THRESHOLD
#define INV_MULMOD_BNM1_THRESHOLD  (4*MULMOD_BNM1_THRESHOLD)
#endif

#ifndef INV_APPR_THRESHOLD
#define INV_APPR_THRESHOLD         INV_NEWTON_THRESHOLD
#endif

#ifndef INV_NEWTON_THRESHOLD
#define INV_NEWTON_THRESHOLD            200
#endif

#ifndef BINV_NEWTON_THRESHOLD
#define BINV_NEWTON_THRESHOLD           300
#endif

#ifndef MU_DIVAPPR_Q_THRESHOLD
#define MU_DIVAPPR_Q_THRESHOLD         2000
#endif

#ifndef MU_DIV_QR_THRESHOLD
#define MU_DIV_QR_THRESHOLD            2000
#endif

#ifndef MUPI_DIV_QR_THRESHOLD
#define MUPI_DIV_QR_THRESHOLD           200
#endif

#ifndef MU_BDIV_Q_THRESHOLD
#define MU_BDIV_Q_THRESHOLD            2000
#endif

#ifndef MU_BDIV_QR_THRESHOLD
#define MU_BDIV_QR_THRESHOLD           2000
#endif

#ifndef MULMOD_BNM1_THRESHOLD
#define MULMOD_BNM1_THRESHOLD            16
#endif

#ifndef SQRMOD_BNM1_THRESHOLD
#define SQRMOD_BNM1_THRESHOLD            16
#endif

#ifndef MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD
#define MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD  (INV_MULMOD_BNM1_THRESHOLD/2)
#endif

#if HAVE_NATIVE_mpn_addmul_2 || HAVE_NATIVE_mpn_redc_2

#ifndef REDC_1_TO_REDC_2_THRESHOLD
#define REDC_1_TO_REDC_2_THRESHOLD       15
#endif
#ifndef REDC_2_TO_REDC_N_THRESHOLD
#define REDC_2_TO_REDC_N_THRESHOLD      100
#endif

#else

#ifndef REDC_1_TO_REDC_N_THRESHOLD
#define REDC_1_TO_REDC_N_THRESHOLD      100
#endif

#endif /* HAVE_NATIVE_mpn_addmul_2 || HAVE_NATIVE_mpn_redc_2 */


/* First k to use for an FFT modF multiply.  A modF FFT is an order
   log(2^k)/log(2^(k-1)) algorithm, so k=3 is merely 1.5 like karatsuba,
   whereas k=4 is 1.33 which is faster than toom3 at 1.485.    */
#define FFT_FIRST_K  4

/* Threshold at which FFT should be used to do a modF NxN -> N multiply. */
#ifndef MUL_FFT_MODF_THRESHOLD
#define MUL_FFT_MODF_THRESHOLD   (MUL_TOOM33_THRESHOLD * 3)
#endif
#ifndef SQR_FFT_MODF_THRESHOLD
#define SQR_FFT_MODF_THRESHOLD   (SQR_TOOM3_THRESHOLD * 3)
#endif

/* Threshold at which FFT should be used to do an NxN -> 2N multiply.  This
   will be a size where FFT is using k=7 or k=8, since an FFT-k used for an
   NxN->2N multiply and not recursing into itself is an order
   log(2^k)/log(2^(k-2)) algorithm, so it'll be at least k=7 at 1.39 which
   is the first better than toom3.  */
#ifndef MUL_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD   (MUL_FFT_MODF_THRESHOLD * 10)
#endif
#ifndef SQR_FFT_THRESHOLD
#define SQR_FFT_THRESHOLD   (SQR_FFT_MODF_THRESHOLD * 10)
#endif

/* Table of thresholds for successive modF FFT "k"s.  The first entry is
   where FFT_FIRST_K+1 should be used, the second FFT_FIRST_K+2,
   etc.  See mpn_fft_best_k(). */
#ifndef MUL_FFT_TABLE
#define MUL_FFT_TABLE							\
  { MUL_TOOM33_THRESHOLD * 4,   /* k=5 */				\
    MUL_TOOM33_THRESHOLD * 8,   /* k=6 */				\
    MUL_TOOM33_THRESHOLD * 16,  /* k=7 */				\
    MUL_TOOM33_THRESHOLD * 32,  /* k=8 */				\
    MUL_TOOM33_THRESHOLD * 96,  /* k=9 */				\
    MUL_TOOM33_THRESHOLD * 288, /* k=10 */				\
    0 }
#endif
#ifndef SQR_FFT_TABLE
#define SQR_FFT_TABLE							\
  { SQR_TOOM3_THRESHOLD * 4,   /* k=5 */				\
    SQR_TOOM3_THRESHOLD * 8,   /* k=6 */				\
    SQR_TOOM3_THRESHOLD * 16,  /* k=7 */				\
    SQR_TOOM3_THRESHOLD * 32,  /* k=8 */				\
    SQR_TOOM3_THRESHOLD * 96,  /* k=9 */				\
    SQR_TOOM3_THRESHOLD * 288, /* k=10 */				\
    0 }
#endif






#if ! defined (invert_limb) && HAVE_NATIVE_mpn_invert_limb
#define invert_limb(invxl,xl)						\
  do {									\
    (invxl) = mpn_invert_limb (xl);					\
  } while (0)
#endif

#ifndef invert_limb
#define invert_limb(invxl,xl)						\
  do {									\
    mp_limb_t _dummy;							\
    ASSERT ((xl) != 0);							\
    udiv_qrnnd (invxl, _dummy, ~(xl), ~CNST_LIMB(0), xl);		\
  } while (0)
#endif

/* n-1 inverts any low zeros and the lowest one bit.  If n&(n-1) leaves zero
   then that lowest one bit must have been the only bit set.  n==0 will
   return true though, so avoid that.  */
#define POW2_P(n)  (((n) & ((n) - 1)) == 0)

/* This is intended for constant THRESHOLDs only, where the compiler
   can completely fold the result.  */
#define LOG2C(n) \
 (((n) >=    0x1) + ((n) >=    0x2) + ((n) >=    0x4) + ((n) >=    0x8) + \
  ((n) >=   0x10) + ((n) >=   0x20) + ((n) >=   0x40) + ((n) >=   0x80) + \
  ((n) >=  0x100) + ((n) >=  0x200) + ((n) >=  0x400) + ((n) >=  0x800) + \
  ((n) >= 0x1000) + ((n) >= 0x2000) + ((n) >= 0x4000) + ((n) >= 0x8000))

#define MP_LIMB_T_MAX      (~ (mp_limb_t) 0)

#define ULONG_HIGHBIT      (ULONG_MAX ^ ((unsigned long) ULONG_MAX >> 1))
#define UINT_HIGHBIT       (UINT_MAX ^ ((unsigned) UINT_MAX >> 1))
#define USHRT_HIGHBIT      (USHRT_MAX ^ ((unsigned short) USHRT_MAX >> 1))
#define GMP_LIMB_HIGHBIT  (MP_LIMB_T_MAX ^ (MP_LIMB_T_MAX >> 1))

#if __GMP_MP_SIZE_T_INT
#define MP_SIZE_T_MAX      INT_MAX
#define MP_SIZE_T_MIN      INT_MIN
#else
#define MP_SIZE_T_MAX      LONG_MAX
#define MP_SIZE_T_MIN      LONG_MIN
#endif

/* mp_exp_t is the same as mp_size_t */
#define MP_EXP_T_MAX   MP_SIZE_T_MAX
#define MP_EXP_T_MIN   MP_SIZE_T_MIN

#define LONG_HIGHBIT       LONG_MIN
#define INT_HIGHBIT        INT_MIN
#define SHRT_HIGHBIT       SHRT_MIN

#define MPN_ZERO(dst, n)						\
  do {									\
    ASSERT ((n) >= 0);							\
    if ((n) != 0)							\
      MPN_FILL (dst, n, CNST_LIMB (0));					\
  } while (0)

#if HAVE_NATIVE_mpn_add_nc
#define mpn_add_nc __MPN(add_nc)
__GMP_DECLSPEC mp_limb_t mpn_add_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#else
ANYCALLER static inline
mp_limb_t
mpn_add_nc (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n, mp_limb_t ci)
{
  mp_limb_t co;
  co = mpn_add_n (rp, up, vp, n);
  co += mpn_add_1 (rp, rp, n, ci);
  return co;
}
#endif

#define mpn_invertappr_itch(n)  (2 * (n))
#define mpn_invert_itch(n)  mpn_invertappr_itch(n)

#if HAVE_NATIVE_mpn_sub_nc
#define mpn_sub_nc __MPN(sub_nc)
__GMP_DECLSPEC mp_limb_t mpn_sub_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#else
ANYCALLER static inline mp_limb_t
mpn_sub_nc (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n, mp_limb_t ci)
{
  mp_limb_t co;
  co = mpn_sub_n (rp, up, vp, n);
  co += mpn_sub_1 (rp, rp, n, ci);
  return co;
}
#endif

#if HAVE_HOST_CPU_FAMILY_power || HAVE_HOST_CPU_FAMILY_powerpc
#define MPN_FILL(dst, n, f)						\
  do {									\
    mp_ptr __dst = (dst) - 1;						\
    mp_size_t __n = (n);						\
    ASSERT (__n > 0);							\
    do									\
      *++__dst = (f);							\
    while (--__n);							\
  } while (0)
#endif

#ifndef MPN_FILL
#define MPN_FILL(dst, n, f)						\
  do {									\
    mp_ptr __dst = (dst);						\
    mp_size_t __n = (n);						\
    ASSERT (__n > 0);							\
    do									\
      *__dst++ = (f);							\
    while (--__n);							\
  } while (0)
#endif

#define GMP_NUMB_HIGHBIT  (CNST_LIMB(1) << (GMP_NUMB_BITS-1))

#if ! defined (__GNUC__) || __GNUC__ < 2
#define __builtin_constant_p(x)   0
#endif


#ifndef SET_STR_PRECOMPUTE_THRESHOLD
#define SET_STR_PRECOMPUTE_THRESHOLD   2000
#endif


#if GMP_NAIL_BITS == 0
#ifndef mpn_incr_u
#define mpn_incr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	while (++(*(__p++)) == 0)					\
	  ;								\
      }									\
    else								\
      {									\
	__x = *__p + (incr);						\
	*__p = __x;							\
	if (__x < (incr))						\
	  while (++(*(++__p)) == 0)					\
	    ;								\
      }									\
  } while (0)
#endif
#ifndef mpn_decr_u
#define mpn_decr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	while ((*(__p++))-- == 0)					\
	  ;								\
      }									\
    else								\
      {									\
	__x = *__p;							\
	*__p = __x - (incr);						\
	if (__x < (incr))						\
	  while ((*(++__p))-- == 0)					\
	    ;								\
      }									\
  } while (0)
#endif
#endif

ANYCALLER static inline int
mpn_jacobi_finish (unsigned bits)
{
  /* (a, b) = (1,0) or (0,1) */
  ASSERT ( (bits & 14) == 0);

  return 1-2*(bits & 1);
}

#define jacobi_table __gmp_jacobi_table

const unsigned char jacobi_table[208] = {
 0, 0, 0, 0, 0,12, 8, 4, 1, 1, 1, 1, 1,13, 9, 5,
 2, 2, 2, 2, 2, 6,10,14, 3, 3, 3, 3, 3, 7,11,15,
 4,16, 6,18, 4, 0,12, 8, 5,17, 7,19, 5, 1,13, 9,
 6,18, 4,16, 6,10,14, 2, 7,19, 5,17, 7,11,15, 3,
 8,10, 9,11, 8, 4, 0,12, 9,11, 8,10, 9, 5, 1,13,
10, 9,11, 8,10,14, 2, 6,11, 8,10, 9,11,15, 3, 7,
12,22,24,20,12, 8, 4, 0,13,23,25,21,13, 9, 5, 1,
25,21,13,23,14, 2, 6,10,24,20,12,22,15, 3, 7,11,
16, 6,18, 4,16,16,16,16,17, 7,19, 5,17,17,17,17,
18, 4,16, 6,18,22,19,23,19, 5,17, 7,19,23,18,22,
20,12,22,24,20,20,20,20,21,13,23,25,21,21,21,21,
22,24,20,12,22,19,23,18,23,25,21,13,23,18,22,19,
24,20,12,22,15, 3, 7,11,25,21,13,23,14, 2, 6,10,
};

#define CALL(expr)							\
  do {									\
    got.flags (data[i].flags);						\
    got.width (data[i].width);						\
    got.precision (data[i].precision);					\
    if (data[i].fill == '\0')						\
      got.fill (' ');							\
    else								\
      got.fill (data[i].fill);						\
									\
    if (! (expr))							\
      {									\
	cout << "\"got\" output error\n";				\
	abort ();							\
      }									\
    if (got.width() != 0)						\
      {									\
	cout << "\"got\" width not reset to 0\n";			\
	abort ();							\
      }									\
									\
  } while (0)

#ifndef MATRIX22_STRASSEN_THRESHOLD
#define MATRIX22_STRASSEN_THRESHOLD 30
#endif

ANYCALLER static inline mp_size_t mpn_sqrmod_bknp1_itch (mp_size_t rn) {
  return rn * 3;
}

ANYCALLER static inline mp_size_t mpn_mulmod_bknp1_itch (mp_size_t rn) {
  return rn << 2;
}

#define mpn_toom22_mul_itch(an, bn) \
  (2 * ((an) + GMP_NUMB_BITS))
#define mpn_toom2_sqr_itch(an) \
  (2 * ((an) + GMP_NUMB_BITS))

/* toom33/toom3: Scratch need is 5an/2 + 10k, k is the recursion depth.
   We use 3an + C, so that we can use a smaller constant.
 */
#define mpn_toom33_mul_itch(an, bn) \
  (3 * (an) + GMP_NUMB_BITS)
#define mpn_toom3_sqr_itch(an) \
  (3 * (an) + GMP_NUMB_BITS)

#define mpn_toom44_mul_itch(an, bn) \
  (3 * (an) + GMP_NUMB_BITS)
#define mpn_toom4_sqr_itch(an) \
  (3 * (an) + GMP_NUMB_BITS)

#define MUL_TOOM6H_MIN							\
  ((MUL_TOOM6H_THRESHOLD > MUL_TOOM44_THRESHOLD) ?			\
    MUL_TOOM6H_THRESHOLD : MUL_TOOM44_THRESHOLD)

#define mpn_toom6_mul_n_itch(n)						\
  (((n) - MUL_TOOM6H_MIN)*2 +						\
   MAX(MUL_TOOM6H_MIN*2 + GMP_NUMB_BITS*6,				\
       mpn_toom44_mul_itch(MUL_TOOM6H_MIN,MUL_TOOM6H_MIN)))


ANYCALLER static inline mp_size_t mpn_toom6h_mul_itch (mp_size_t an, mp_size_t bn) {
  mp_size_t estimatedN;
  estimatedN = (an + bn) / (size_t) 10 + 1;
  return mpn_toom6_mul_n_itch (estimatedN * 6);
}
#if 0
#define mpn_fft_mul mpn_mul_fft_full
#else
#define mpn_fft_mul mpn_nussbaumer_mul
#endif

#define MUL_TOOM8H_MIN							\
  ((MUL_TOOM8H_THRESHOLD > MUL_TOOM6H_MIN) ?				\
    MUL_TOOM8H_THRESHOLD : MUL_TOOM6H_MIN)
#define mpn_toom8_mul_n_itch(n)						\
  ((((n)*15)>>3) - ((MUL_TOOM8H_MIN*15)>>3) +				\
   MAX(((MUL_TOOM8H_MIN*15)>>3) + GMP_NUMB_BITS*6,			\
       mpn_toom6_mul_n_itch(MUL_TOOM8H_MIN)))

#define mpn_toom8_sqr_itch(n)						\
  ((((n)*15)>>3) - ((SQR_TOOM8_THRESHOLD*15)>>3) +			\
   MAX(((SQR_TOOM8_THRESHOLD*15)>>3) + GMP_NUMB_BITS*6,			\
       mpn_toom6_sqr_itch(SQR_TOOM8_THRESHOLD)))

ANYCALLER static inline mp_size_t
mpn_toom8h_mul_itch (mp_size_t an, mp_size_t bn) {
  mp_size_t estimatedN;
  estimatedN = (an + bn) / (size_t) 14 + 1;
  return mpn_toom8_mul_n_itch (estimatedN * 8);
}



ANYCALLER static inline mp_size_t
mpn_toom32_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (2 * an >= 3 * bn ? (an - 1) / (size_t) 3 : (bn - 1) >> 1);
  mp_size_t itch = 2 * n + 1;

  return itch;
}

ANYCALLER static inline mp_size_t
mpn_toom42_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = an >= 2 * bn ? (an + 3) >> 2 : (bn + 1) >> 1;
  return 6 * n + 3;
}

ANYCALLER static inline mp_size_t
mpn_toom43_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (3 * an >= 4 * bn ? (an - 1) >> 2 : (bn - 1) / (size_t) 3);

  return 6*n + 4;
}

ANYCALLER static inline mp_size_t
mpn_toom52_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (2 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) >> 1);
  return 6*n + 4;
}

ANYCALLER static inline mp_size_t
mpn_toom53_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (3 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 3);
  return 10 * n + 10;
}

ANYCALLER static inline mp_size_t
mpn_toom62_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (an >= 3 * bn ? (an - 1) / (size_t) 6 : (bn - 1) >> 1);
  return 10 * n + 10;
}

ANYCALLER static inline mp_size_t
mpn_toom63_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (an >= 2 * bn ? (an - 1) / (size_t) 6 : (bn - 1) / (size_t) 3);
  return 9 * n + 3;
}

ANYCALLER static inline mp_size_t
mpn_toom54_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (4 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 4);
  return 9 * n + 3;
}

/* let S(n) = space required for input size n,
   then S(n) = 3 floor(n/2) + 1 + S(floor(n/2)).   */
#define mpn_toom42_mulmid_itch(n) \
  (3 * (n) + GMP_NUMB_BITS)


#if MOD_BKNP1_ONLY3
#define MPN_SQRMOD_BKNP1_USABLE(rn, k, mn)				\
  MPN_MULMOD_BKNP1_USABLE(rn, k, mn)
#else
#define MPN_SQRMOD_BKNP1_USABLE(rn, k, mn)				\
  (((GMP_NUMB_BITS % 8 == 0) && ((mn) >= 27) && ((rn) > 24) &&		\
    (((rn) % ((k) = 3) == 0) ||						\
     (((GMP_NUMB_BITS % 16 != 0) || (((mn) >= 55) && ((rn) > 50))) &&	\
      (((GMP_NUMB_BITS % 16 == 0) && ((rn) % ((k) = 5) == 0)) ||	\
       (((mn) >= 56) &&							\
	(((rn) % ((k) = 7) == 0) ||					\
	 ((GMP_NUMB_BITS % 16 == 0) && ((mn) >= 143) && ((rn) >= 128) && \
	  ((MOD_BKNP1_USE11 && ((rn) % ((k) = 11) == 0)) ||		\
	   ((rn) % ((k) = 13) == 0) ||					\
	   ((GMP_NUMB_BITS % 32 == 0) && ((mn) >= 272) && ((rn) >= 256) && \
	    ((rn) % ((k) = 17) == 0)					\
	    ))))))))) ||						\
   ((GMP_NUMB_BITS % 16 != 0) && MOD_BKNP1_USE11 &&			\
    ((mn) >= 143) && ((rn) >= 128) && ((rn) % ((k) = 11) == 0)) )
#endif


ANYCALLER static inline mp_size_t
mpn_sqrmod_bnm1_itch (mp_size_t rn, mp_size_t an) {
  mp_size_t n, itch;
  n = rn >> 1;
  itch = rn + 3 +
    (an > n ? an : 0);
  return itch;
}


/* toom33/toom3: Scratch need is 8an/3 + 13k, k is the recursion depth.
   We use 3an + C, so that we can use a smaller constant.
 */
#define mpn_toom44_mul_itch(an, bn) \
  (3 * (an) + GMP_NUMB_BITS)
#define mpn_toom4_sqr_itch(an) \
  (3 * (an) + GMP_NUMB_BITS)

#define mpn_toom6_sqr_itch(n)						\
  (((n) - SQR_TOOM6_THRESHOLD)*2 +					\
   MAX(SQR_TOOM6_THRESHOLD*2 + GMP_NUMB_BITS*6,				\
       mpn_toom4_sqr_itch(SQR_TOOM6_THRESHOLD)))

#define MUL_TOOM6H_MIN							\
  ((MUL_TOOM6H_THRESHOLD > MUL_TOOM44_THRESHOLD) ?			\
    MUL_TOOM6H_THRESHOLD : MUL_TOOM44_THRESHOLD)

#ifndef MOD_BKNP1_USE11
#define MOD_BKNP1_USE11 ((GMP_NUMB_BITS % 8 != 0) && (GMP_NUMB_BITS % 2 == 0))
#endif
#ifndef MOD_BKNP1_ONLY3
#define MOD_BKNP1_ONLY3 0
#endif

#if MOD_BKNP1_ONLY3
#define MPN_MULMOD_BKNP1_USABLE(rn, k, mn)				\
  ((GMP_NUMB_BITS % 8 == 0) && ((mn) >= 18) && ((rn) > 16) &&		\
   (((rn) % ((k) = 3) == 0)))
#else
#define MPN_MULMOD_BKNP1_USABLE(rn, k, mn)				\
  (((GMP_NUMB_BITS % 8 == 0) && ((mn) >= 18) && ((rn) > 16) &&		\
    (((rn) % ((k) = 3) == 0) ||						\
     (((GMP_NUMB_BITS % 16 != 0) || (((mn) >= 35) && ((rn) >= 32))) &&	\
      (((GMP_NUMB_BITS % 16 == 0) && ((rn) % ((k) = 5) == 0)) ||	\
       (((mn) >= 49) &&							\
	(((rn) % ((k) = 7) == 0) ||					\
	 ((GMP_NUMB_BITS % 16 == 0) && ((mn) >= 104) && ((rn) >= 64) &&	\
	  ((MOD_BKNP1_USE11 && ((rn) % ((k) = 11) == 0)) ||		\
	   ((rn) % ((k) = 13) == 0) ||					\
	   ((GMP_NUMB_BITS % 32 == 0) && ((mn) >= 136) && ((rn) >= 128) && \
	    ((rn) % ((k) = 17) == 0)					\
	    ))))))))) ||						\
  ((GMP_NUMB_BITS % 16 != 0) && MOD_BKNP1_USE11 &&			\
   ((mn) >= 104) && ((rn) >= 64) && ((rn) % ((k) = 11) == 0)) )
#endif

ANYCALLER static inline unsigned
mpn_jacobi_update (unsigned bits, unsigned denominator, unsigned q)
{
  /* FIXME: Could halve table size by not including the e bit in the
   * index, and instead xor when updating. Then the lookup would be
   * like
   *
   *   bits ^= table[((bits & 30) << 2) + (denominator << 2) + q];
   */

  ASSERT (bits < 26);
  ASSERT (denominator < 2);
  ASSERT (q < 4);

  /* For almost all calls, denominator is constant and quite often q
     is constant too. So use addition rather than or, so the compiler
     can put the constant part can into the offset of an indexed
     addressing instruction.

     With constant denominator, the below table lookup is compiled to

       C Constant q = 1, constant denominator = 1
       movzbl table+5(%eax,8), %eax

     or

       C q in %edx, constant denominator = 1
       movzbl table+4(%edx,%eax,8), %eax

     One could maintain the state preshifted 3 bits, to save a shift
     here, but at least on x86, that's no real saving.
  */
  return jacobi_table[(bits << 3) + (denominator << 2) + q];
}

#define gmp_udiv_qrnnd_preinv(q, r, nh, nl, d, di)			\
  do {									\
    mp_limb_t _qh, _ql, _r, _mask;					\
    gmp_umul_ppmm (_qh, _ql, (nh), (di));				\
    gmp_add_ssaaaa (_qh, _ql, _qh, _ql, (nh) + 1, (nl));		\
    _r = (nl) - gmp_umullo_limb (_qh, (d));				\
    _mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */		\
    _qh += _mask;							\
    _r += _mask & (d);							\
    if (_r >= (d))							\
      {									\
	_r -= (d);							\
	_qh++;								\
      }									\
									\
    (r) = _r;								\
    (q) = _qh;								\
  } while (0)

/* Use (4.0 * ...) instead of (2.0 * ...) to work around buggy compilers
   that don't convert ulong->double correctly (eg. SunOS 4 native cc).  */
#define MP_BASE_AS_DOUBLE (4.0 * ((mp_limb_t) 1 << (GMP_NUMB_BITS - 2)))
/* Maximum number of limbs it will take to store any `double'.
   We assume doubles have 53 mantissa bits.  */
#define LIMBS_PER_DOUBLE ((53 + GMP_NUMB_BITS - 2) / GMP_NUMB_BITS + 1)

#define gmp_udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv)		\
  do {									\
    mp_limb_t _q0, _t1, _t0, _mask;					\
    gmp_umul_ppmm ((q), _q0, (n2), (dinv));				\
    gmp_add_ssaaaa ((q), _q0, (q), _q0, (n2), (n1));			\
									\
    /* Compute the two most significant limbs of n - q'd */		\
    (r1) = (n1) - gmp_umullo_limb ((d1), (q));				\
    gmp_sub_ddmmss ((r1), (r0), (r1), (n0), (d1), (d0));		\
    gmp_umul_ppmm (_t1, _t0, (d0), (q));				\
    gmp_sub_ddmmss ((r1), (r0), (r1), (r0), _t1, _t0);			\
    (q)++;								\
									\
    /* Conditionally adjust q and the remainders */			\
    _mask = - (mp_limb_t) ((r1) >= _q0);				\
    (q) += _mask;							\
    gmp_add_ssaaaa ((r1), (r0), (r1), (r0), _mask & (d1), _mask & (d0)); \
    if ((r1) >= (d1))							\
      {									\
	if ((r1) > (d1) || (r0) >= (d0))				\
	  {								\
	    (q)++;							\
	    gmp_sub_ddmmss ((r1), (r0), (r1), (r0), (d1), (d0));	\
	  }								\
      }									\
  } while (0)

  #define udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv)		\
  do {									\
    mp_limb_t _q0, _t1, _t0, _mask;					\
    umul_ppmm ((q), _q0, (n2), (dinv));					\
    add_ssaaaa ((q), _q0, (q), _q0, (n2), (n1));			\
									\
    /* Compute the two most significant limbs of n - q'd */		\
    (r1) = (n1) - (d1) * (q);						\
    sub_ddmmss ((r1), (r0), (r1), (n0), (d1), (d0));			\
    umul_ppmm (_t1, _t0, (d0), (q));					\
    sub_ddmmss ((r1), (r0), (r1), (r0), _t1, _t0);			\
    (q)++;								\
									\
    /* Conditionally adjust q and the remainders */			\
    _mask = - (mp_limb_t) ((r1) >= _q0);				\
    (q) += _mask;							\
    add_ssaaaa ((r1), (r0), (r1), (r0), _mask & (d1), _mask & (d0));	\
    if (UNLIKELY ((r1) >= (d1)))					\
      {									\
	if ((r1) > (d1) || (r0) >= (d0))				\
	  {								\
	    (q)++;							\
	    sub_ddmmss ((r1), (r0), (r1), (r0), (d1), (d0));		\
	  }								\
      }									\
  } while (0)

/* gmp_uint_least32_t is an unsigned integer type with at least 32 bits. */
#if HAVE_UINT_LEAST32_T
typedef uint_least32_t      gmp_uint_least32_t;
#else
#if SIZEOF_UNSIGNED_SHORT >= 4
typedef unsigned short      gmp_uint_least32_t;
#else
#if SIZEOF_UNSIGNED >= 4
typedef unsigned            gmp_uint_least32_t;
#else
typedef unsigned long       gmp_uint_least32_t;
#endif
#endif
#endif











enum toom6_flags {toom6_all_pos = 0, toom6_vm1_neg = 1, toom6_vm2_neg = 2};
enum toom7_flags { toom7_w1_neg = 1, toom7_w3_neg = 2 };

#define GMP_ERROR(code)   __gmp_exception (code)
#define DIVIDE_BY_ZERO    __gmp_divide_by_zero ()
#define SQRT_OF_NEGATIVE  __gmp_sqrt_of_negative ()
#define MPZ_OVERFLOW      __gmp_overflow_in_mpz ()

#define   mpn_bdiv_dbm1(dst, src, size, divisor) \
  mpn_bdiv_dbm1c (dst, src, size, divisor, __GMP_CAST (mp_limb_t, 0))


// mp_bpl.c

const int __gmp_0 = 0;
int __gmp_junk;

// EOF mp_bpl.c

// errno.c

int gmp_errno = 0;


/* Use SIGFPE on systems which have it. Otherwise, deliberate divide
   by zero, which triggers an exception on most systems. On those
   where it doesn't, for example power and powerpc, use abort instead. */
void
__gmp_exception (int error_bit)
{
  gmp_errno |= error_bit;
#ifdef SIGFPE
  raise (SIGFPE);
#else
  __gmp_junk = 10 / __gmp_0;
#endif
  abort ();
}


/* These functions minimize the amount of code required in functions raising
   exceptions.  Since they're "noreturn" and don't take any parameters, a
   test and call might even come out as a simple conditional jump.  */
void
__gmp_sqrt_of_negative (void)
{
  __gmp_exception (GMP_ERROR_SQRT_OF_NEGATIVE);
}
void
__gmp_divide_by_zero (void)
{
  __gmp_exception (GMP_ERROR_DIVISION_BY_ZERO);
}
void
__gmp_overflow_in_mpz (void)
{
  __gmp_exception (GMP_ERROR_MPZ_OVERFLOW);
}

// EOF errno.c










/* END MPN INTERNALS PORTING VOMIT */



