//"Wrapper" header meant to mimic gmp.h
#pragma once
#include "gmp.h"
#include "Definitions.cuh"

// GPGMP's equivalent of __GMP_DECLSPEC.
#define __GPGMP_DECLSPEC
#define __GPGMP_CALLERTYPE ANYCALLER
#define __GPGMP_MPN(x) __gpgmpn_##x

/************ Low level positive-integer (i.e. N) routines.  ************/

namespace gpgmp {
    namespace mpnRoutines {


/* This is ugly, but we need to make user calls reach the prefixed function. */
#define __GMP_FORCE_gpmpn_add 1
#define __GMP_FORCE_gpmpn_cmp 1
#define __GMP_FORCE_gpmpn_neg 1
#define __GMP_FORCE_gpmpn_sub_1 1
#define __GMP_FORCE_gpmpn_sub 1
#define __GMP_FORCE_gpmpn_zero_p 1
#define __GMP_FORCE_gpmpn_add_1 1

#define gpmpn_add __GPGMP_MPN(add)
#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_add)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#endif

#define gpmpn_add_1 __GPGMP_MPN(add_1)
#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_add_1)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t) __GMP_NOTHROW;
#endif

#define gpmpn_add_n __GPGMP_MPN(add_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_addmul_1 __GPGMP_MPN(addmul_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_cmp __GPGMP_MPN(cmp)
#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_cmp)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_cmp(mp_srcptr, mp_srcptr, mp_size_t)
//__GMP_NOTHROW __GMP_ATTRIBUTE_PURE;
#endif

#define gpmpn_zero_p __GPGMP_MPN(zero_p)
#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_zero_p)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_zero_p(mp_srcptr, mp_size_t)
//__GMP_NOTHROW __GMP_ATTRIBUTE_PURE;
#endif

#define gpmpn_divexact_1 __GPGMP_MPN(divexact_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_divexact_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_divexact_by3(dst, src, size) \
    gpmpn_divexact_by3c(dst, src, size, __GMP_CAST(mp_limb_t, 0))

#define gpmpn_divexact_by3c __GPGMP_MPN(divexact_by3c)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divexact_by3c(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_divmod_1(qp, np, nsize, dlimb) \
    gpmpn_divrem_1(qp, __GMP_CAST(mp_size_t, 0), np, nsize, dlimb)

#define gpmpn_divrem __GPGMP_MPN(divrem)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divrem(mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_divrem_1 __GPGMP_MPN(divrem_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divrem_1(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_divrem_2 __GPGMP_MPN(divrem_2)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divrem_2(mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr);

#define gpmpn_div_qr_1 __GPGMP_MPN(div_qr_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_div_qr_1(mp_ptr, mp_limb_t *, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_div_qr_2 __GPGMP_MPN(div_qr_2)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_div_qr_2(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_gcd __GPGMP_MPN(gcd)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_gcd(mp_ptr, mp_ptr, mp_size_t, mp_ptr, mp_size_t);

#define gpmpn_gcd_11 __GPGMP_MPN(gcd_11)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_gcd_11(mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_gcd_1 __GPGMP_MPN(gcd_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_gcd_1(mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_gcdext_1 __GPGMP_MPN(gcdext_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_gcdext_1(mp_limb_signed_t *, mp_limb_signed_t *, mp_limb_t, mp_limb_t);

#define gpmpn_gcdext __GPGMP_MPN(gcdext)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_gcdext(mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_size_t, mp_ptr, mp_size_t);

#define gpmpn_get_str __GPGMP_MPN(get_str)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpn_get_str(unsigned char *, int, mp_ptr, mp_size_t);

#define gpmpn_hamdist __GPGMP_MPN(hamdist)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_hamdist(mp_srcptr, mp_srcptr, mp_size_t)
__GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

#define gpmpn_lshift __GPGMP_MPN(lshift)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_lshift(mp_ptr, mp_srcptr, mp_size_t, unsigned int);

#define gpmpn_mod_1 __GPGMP_MPN(mod_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_1(mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_mul __GPGMP_MPN(mul)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_mul_1 __GPGMP_MPN(mul_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_mul_n __GPGMP_MPN(mul_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mul_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_sqr __GPGMP_MPN(sqr)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqr(mp_ptr, mp_srcptr, mp_size_t);

#define gpmpn_neg __GPGMP_MPN(neg)
#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_neg)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_neg(mp_ptr, mp_srcptr, mp_size_t);
#endif

#define gpmpn_com __GPGMP_MPN(com)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_com(mp_ptr, mp_srcptr, mp_size_t);

#define gpmpn_perfect_square_p __GPGMP_MPN(perfect_square_p)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_perfect_square_p(mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_perfect_power_p __GPGMP_MPN(perfect_power_p)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_perfect_power_p(mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_popcount __GPGMP_MPN(popcount)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_popcount(mp_srcptr, mp_size_t)
__GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

#define gpmpn_pow_1 __GPGMP_MPN(pow_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_pow_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);

/* undocumented now, but retained here for upward compatibility */
#define gpmpn_preinv_mod_1 __GPGMP_MPN(preinv_mod_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_preinv_mod_1(mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_random __GPGMP_MPN(random)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_random(mp_ptr, mp_size_t);

#define gpmpn_random2 __GPGMP_MPN(random2)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_random2(mp_ptr, mp_size_t);

#define gpmpn_rshift __GPGMP_MPN(rshift)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_rshift(mp_ptr, mp_srcptr, mp_size_t, unsigned int);

#define gpmpn_scan0 __GPGMP_MPN(scan0)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_scan0(mp_srcptr, mp_bitcnt_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_scan1 __GPGMP_MPN(scan1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_scan1(mp_srcptr, mp_bitcnt_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_set_str __GPGMP_MPN(set_str)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_set_str(mp_ptr, const unsigned char *, size_t, int);

#define gpmpn_sizeinbase __GPGMP_MPN(sizeinbase)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpn_sizeinbase(mp_srcptr, mp_size_t, int);

#define gpmpn_sqrtrem __GPGMP_MPN(sqrtrem)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sqrtrem(mp_ptr, mp_ptr, mp_srcptr, mp_size_t);

#define gpmpn_sub __GPGMP_MPN(sub)
#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_sub)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#endif

#define gpmpn_sub_1 __GPGMP_MPN(sub_1)
#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_sub_1)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t) __GMP_NOTHROW;
#endif

#define gpmpn_sub_n __GPGMP_MPN(sub_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_submul_1 __GPGMP_MPN(submul_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_submul_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_tdiv_qr __GPGMP_MPN(tdiv_qr)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_tdiv_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_and_n __GPGMP_MPN(and_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_and_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_andn_n __GPGMP_MPN(andn_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_andn_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_nand_n __GPGMP_MPN(nand_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_nand_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_ior_n __GPGMP_MPN(ior_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_ior_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_iorn_n __GPGMP_MPN(iorn_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_iorn_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_nior_n __GPGMP_MPN(nior_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_nior_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_xor_n __GPGMP_MPN(xor_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_xor_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_xnor_n __GPGMP_MPN(xnor_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_xnor_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_copyi __GPGMP_MPN(copyi)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_copyi(mp_ptr, mp_srcptr, mp_size_t);
#define gpmpn_copyd __GPGMP_MPN(copyd)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_copyd(mp_ptr, mp_srcptr, mp_size_t);
#define gpmpn_zero __GPGMP_MPN(zero)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_zero(mp_ptr, mp_size_t);

#define gpmpn_cnd_add_n __GPGMP_MPN(cnd_add_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_cnd_add_n(mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_cnd_sub_n __GPGMP_MPN(cnd_sub_n)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_cnd_sub_n(mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_sec_add_1 __GPGMP_MPN(sec_add_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sec_add_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
#define gpmpn_sec_add_1_itch __GPGMP_MPN(sec_add_1_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_add_1_itch(mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_sec_sub_1 __GPGMP_MPN(sec_sub_1)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sec_sub_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
#define gpmpn_sec_sub_1_itch __GPGMP_MPN(sec_sub_1_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_sub_1_itch(mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_cnd_swap __GPGMP_MPN(cnd_swap)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_cnd_swap(mp_limb_t, volatile mp_limb_t *, volatile mp_limb_t *, mp_size_t);

#define gpmpn_sec_mul __GPGMP_MPN(sec_mul)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_sec_mul_itch __GPGMP_MPN(sec_mul_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_mul_itch(mp_size_t, mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_sec_sqr __GPGMP_MPN(sec_sqr)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_sqr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_sec_sqr_itch __GPGMP_MPN(sec_sqr_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_sqr_itch(mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_sec_powm __GPGMP_MPN(sec_powm)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_powm(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_bitcnt_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_sec_powm_itch __GPGMP_MPN(sec_powm_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_powm_itch(mp_size_t, mp_bitcnt_t, mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_sec_tabselect __GPGMP_MPN(sec_tabselect)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_tabselect(volatile mp_limb_t *, volatile const mp_limb_t *, mp_size_t, mp_size_t, mp_size_t);

#define gpmpn_sec_div_qr __GPGMP_MPN(sec_div_qr)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sec_div_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_sec_div_qr_itch __GPGMP_MPN(sec_div_qr_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_div_qr_itch(mp_size_t, mp_size_t) __GMP_ATTRIBUTE_PURE;
#define gpmpn_sec_div_r __GPGMP_MPN(sec_div_r)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_div_r(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_sec_div_r_itch __GPGMP_MPN(sec_div_r_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_div_r_itch(mp_size_t, mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_sec_invert __GPGMP_MPN(sec_invert)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_sec_invert(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_bitcnt_t, mp_ptr);
#define gpmpn_sec_invert_itch __GPGMP_MPN(sec_invert_itch)
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_invert_itch(mp_size_t) __GMP_ATTRIBUTE_PURE;

/**************** mpn inlines ****************/

/* The comments with __GMPN_ADD_1 below apply here too.

   The test for FUNCTION returning 0 should predict well.  If it's assumed
   {yp,ysize} will usually have a random number of bits then the high limb
   won't be full and a carry out will occur a good deal less than 50% of the
   time.

   ysize==0 isn't a documented feature, but is used internally in a few
   places.

   Producing cout last stops it using up a register during the main part of
   the calculation, though gcc (as of 3.0) on an "if (gpmpn_add (...))"
   doesn't seem able to move the true and false legs of the conditional up
   to the two places cout is generated.  */

#define __GPGMPN_AORS(cout, wp, xp, xsize, yp, ysize, FUNCTION, TEST)  \
    do                                                                 \
    {                                                                  \
        mp_size_t __gmp_i;                                             \
        mp_limb_t __gmp_x;                                             \
                                                                       \
        /* ASSERT ((ysize) >= 0); */                                   \
        /* ASSERT ((xsize) >= (ysize)); */                             \
        /* ASSERT (MPN_SAME_OR_SEPARATE2_P (wp, xsize, xp, xsize)); */ \
        /* ASSERT (MPN_SAME_OR_SEPARATE2_P (wp, xsize, yp, ysize)); */ \
                                                                       \
        __gmp_i = (ysize);                                             \
        if (__gmp_i != 0)                                              \
        {                                                              \
            if (FUNCTION(wp, xp, yp, __gmp_i))                         \
            {                                                          \
                do                                                     \
                {                                                      \
                    if (__gmp_i >= (xsize))                            \
                    {                                                  \
                        (cout) = 1;                                    \
                        goto __gmp_done;                               \
                    }                                                  \
                    __gmp_x = (xp)[__gmp_i];                           \
                } while (TEST);                                        \
            }                                                          \
        }                                                              \
        if ((wp) != (xp))                                              \
            __GMPN_COPY_REST(wp, xp, xsize, __gmp_i);                  \
        (cout) = 0;                                                    \
    __gmp_done:;                                                       \
    } while (0)

#define __GPGMPN_ADD(cout, wp, xp, xsize, yp, ysize)           \
    __GPGMPN_AORS(cout, wp, xp, xsize, yp, ysize, gpmpn_add_n, \
                  (((wp)[__gmp_i++] = (__gmp_x + 1) & GMP_NUMB_MASK) == 0))
#define __GPGMPN_SUB(cout, wp, xp, xsize, yp, ysize)           \
    __GPGMPN_AORS(cout, wp, xp, xsize, yp, ysize, gpmpn_sub_n, \
                  (((wp)[__gmp_i++] = (__gmp_x - 1) & GMP_NUMB_MASK), __gmp_x == 0))

/* The use of __gmp_i indexing is designed to ensure a compile time src==dst
   remains nice and clear to the compiler, so that __GMPN_COPY_REST can
   disappear, and the load/add/store gets a chance to become a
   read-modify-write on CISC CPUs.

   Alternatives:

   Using a pair of pointers instead of indexing would be possible, but gcc
   isn't able to recognise compile-time src==dst in that case, even when the
   pointers are incremented more or less together.  Other compilers would
   very likely have similar difficulty.

   gcc could use "if (__builtin_constant_p(src==dst) && src==dst)" or
   similar to detect a compile-time src==dst.  This works nicely on gcc
   2.95.x, it's not good on gcc 3.0 where __builtin_constant_p(p==p) seems
   to be always false, for a pointer p.  But the current code form seems
   good enough for src==dst anyway.

   gcc on x86 as usual doesn't give particularly good flags handling for the
   carry/borrow detection.  It's tempting to want some multi instruction asm
   blocks to help it, and this was tried, but in truth there's only a few
   instructions to save and any gain is all too easily lost by register
   juggling setting up for the asm.  */

#if GMP_NAIL_BITS == 0
#define __GPGMPN_AORS_1(cout, dst, src, n, v, OP, CB)           \
    do                                                          \
    {                                                           \
        mp_size_t __gmp_i;                                      \
        mp_limb_t __gmp_x, __gmp_r;                             \
                                                                \
        /* ASSERT ((n) >= 1); */                                \
        /* ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, n)); */    \
                                                                \
        __gmp_x = (src)[0];                                     \
        __gmp_r = __gmp_x OP(v);                                \
        (dst)[0] = __gmp_r;                                     \
        if (CB(__gmp_r, __gmp_x, (v)))                          \
        {                                                       \
            (cout) = 1;                                         \
            for (__gmp_i = 1; __gmp_i < (n);)                   \
            {                                                   \
                __gmp_x = (src)[__gmp_i];                       \
                __gmp_r = __gmp_x OP 1;                         \
                (dst)[__gmp_i] = __gmp_r;                       \
                ++__gmp_i;                                      \
                if (!CB(__gmp_r, __gmp_x, 1))                   \
                {                                               \
                    if ((src) != (dst))                         \
                        __GMPN_COPY_REST(dst, src, n, __gmp_i); \
                    (cout) = 0;                                 \
                    break;                                      \
                }                                               \
            }                                                   \
        }                                                       \
        else                                                    \
        {                                                       \
            if ((src) != (dst))                                 \
                __GMPN_COPY_REST(dst, src, n, 1);               \
            (cout) = 0;                                         \
        }                                                       \
    } while (0)
#endif

#if GMP_NAIL_BITS >= 1
#define __GPGMPN_AORS_1(cout, dst, src, n, v, OP, CB)           \
    do                                                          \
    {                                                           \
        mp_size_t __gmp_i;                                      \
        mp_limb_t __gmp_x, __gmp_r;                             \
                                                                \
        /* ASSERT ((n) >= 1); */                                \
        /* ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, n)); */    \
                                                                \
        __gmp_x = (src)[0];                                     \
        __gmp_r = __gmp_x OP(v);                                \
        (dst)[0] = __gmp_r & GMP_NUMB_MASK;                     \
        if (__gmp_r >> GMP_NUMB_BITS != 0)                      \
        {                                                       \
            (cout) = 1;                                         \
            for (__gmp_i = 1; __gmp_i < (n);)                   \
            {                                                   \
                __gmp_x = (src)[__gmp_i];                       \
                __gmp_r = __gmp_x OP 1;                         \
                (dst)[__gmp_i] = __gmp_r & GMP_NUMB_MASK;       \
                ++__gmp_i;                                      \
                if (__gmp_r >> GMP_NUMB_BITS == 0)              \
                {                                               \
                    if ((src) != (dst))                         \
                        __GMPN_COPY_REST(dst, src, n, __gmp_i); \
                    (cout) = 0;                                 \
                    break;                                      \
                }                                               \
            }                                                   \
        }                                                       \
        else                                                    \
        {                                                       \
            if ((src) != (dst))                                 \
                __GMPN_COPY_REST(dst, src, n, 1);               \
            (cout) = 0;                                         \
        }                                                       \
    } while (0)
#endif

#define __GPGMPN_ADDCB(r, x, y) ((r) < (y))
#define __GPGMPN_SUBCB(r, x, y) ((x) < (y))

#define __GPGMPN_ADD_1(cout, dst, src, n, v) \
    __GPGMPN_AORS_1(cout, dst, src, n, v, +, __GPGMPN_ADDCB)
#define __GPGMPN_SUB_1(cout, dst, src, n, v) \
    __GPGMPN_AORS_1(cout, dst, src, n, v, -, __GPGMPN_SUBCB)

/* Compare {xp,size} and {yp,size}, setting "result" to positive, zero or
   negative.  size==0 is allowed.  On random data usually only one limb will
   need to be examined to get a result, so it's worth having it inline.  */
#define __GPGMPN_CMP(result, xp, yp, size)                                \
    do                                                                    \
    {                                                                     \
        mp_size_t __gmp_i;                                                \
        mp_limb_t __gmp_x, __gmp_y;                                       \
                                                                          \
        /* ASSERT ((size) >= 0); */                                       \
                                                                          \
        (result) = 0;                                                     \
        __gmp_i = (size);                                                 \
        while (--__gmp_i >= 0)                                            \
        {                                                                 \
            __gmp_x = (xp)[__gmp_i];                                      \
            __gmp_y = (yp)[__gmp_i];                                      \
            if (__gmp_x != __gmp_y)                                       \
            {                                                             \
                /* Cannot use __gmp_x - __gmp_y, may overflow an "int" */ \
                (result) = (__gmp_x > __gmp_y ? 1 : -1);                  \
                break;                                                    \
            }                                                             \
        }                                                                 \
    } while (0)

#if defined(__GMPN_COPY) && !defined(__GMPN_COPY_REST)
#define __GMPN_COPY_REST(dst, src, size, start)                          \
    do                                                                   \
    {                                                                    \
        /* ASSERT ((start) >= 0); */                                     \
        /* ASSERT ((start) <= (size)); */                                \
        __GMPN_COPY((dst) + (start), (src) + (start), (size) - (start)); \
    } while (0)
#endif

/* Copy {src,size} to {dst,size}, starting at "start".  This is designed to
   keep the indexing dst[j] and src[j] nice and simple for __GMPN_ADD_1,
   __GMPN_ADD, etc.  */
#if !defined(__GMPN_COPY_REST)
#define __GMPN_COPY_REST(dst, src, size, start)                 \
    do                                                          \
    {                                                           \
        mp_size_t __gmp_j;                                      \
        /* ASSERT ((size) >= 0); */                             \
        /* ASSERT ((start) >= 0); */                            \
        /* ASSERT ((start) <= (size)); */                       \
        /* ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, size)); */ \
        __GMP_CRAY_Pragma("_CRI ivdep");                        \
        for (__gmp_j = (start); __gmp_j < (size); __gmp_j++)    \
            (dst)[__gmp_j] = (src)[__gmp_j];                    \
    } while (0)
#endif

/* Enhancement: Use some of the smarter code from gmp-impl.h.  Maybe use
   gpmpn_copyi if there's a native version, and if we don't mind demanding
   binary compatibility for it (on targets which use it).  */

#if !defined(__GMPN_COPY)
#define __GMPN_COPY(dst, src, size) __GMPN_COPY_REST(dst, src, size, 0)
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_add)
#if !defined(__GMP_FORCE_gpmpn_add)
__GMP_EXTERN_INLINE
#endif
ANYCALLER static inline  mp_limb_t
gpmpn_add(mp_ptr __gmp_wp, mp_srcptr __gmp_xp, mp_size_t __gmp_xsize, mp_srcptr __gmp_yp, mp_size_t __gmp_ysize)
{
    mp_limb_t __gmp_c;
    __GMPN_ADD(__gmp_c, __gmp_wp, __gmp_xp, __gmp_xsize, __gmp_yp, __gmp_ysize);
    return __gmp_c;
}
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_add_1)
#if !defined(__GMP_FORCE_gpmpn_add_1)
__GMP_EXTERN_INLINE
#endif
ANYCALLER static inline  mp_limb_t
gpmpn_add_1(mp_ptr __gmp_dst, mp_srcptr __gmp_src, mp_size_t __gmp_size, mp_limb_t __gmp_n) __GMP_NOTHROW
{
    mp_limb_t __gmp_c;
    __GMPN_ADD_1(__gmp_c, __gmp_dst, __gmp_src, __gmp_size, __gmp_n);
    return __gmp_c;
}
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_cmp)
#if !defined(__GMP_FORCE_gpmpn_cmp)
__GMP_EXTERN_INLINE
#endif
ANYCALLER static inline  int
gpmpn_cmp(mp_srcptr __gmp_xp, mp_srcptr __gmp_yp, mp_size_t __gmp_size) __GMP_NOTHROW
{
    int __gmp_result;
    __GMPN_CMP(__gmp_result, __gmp_xp, __gmp_yp, __gmp_size);
    return __gmp_result;
}
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_zero_p)
#if !defined(__GMP_FORCE_gpmpn_zero_p)
__GMP_EXTERN_INLINE
#endif
ANYCALLER static inline  int gpmpn_zero_p(mp_srcptr __gmp_p, mp_size_t __gmp_n) __GMP_NOTHROW
{
    /* if (__GMP_LIKELY (__gmp_n > 0)) */
    do
    {
        if (__gmp_p[--__gmp_n] != 0)
            return 0;
    } while (__gmp_n != 0);
    return 1;
}
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_sub)
#if !defined(__GMP_FORCE_gpmpn_sub)
__GMP_EXTERN_INLINE
#endif
ANYCALLER static inline  mp_limb_t
gpmpn_sub(mp_ptr __gmp_wp, mp_srcptr __gmp_xp, mp_size_t __gmp_xsize, mp_srcptr __gmp_yp, mp_size_t __gmp_ysize)
{
    mp_limb_t __gmp_c;
    __GMPN_SUB(__gmp_c, __gmp_wp, __gmp_xp, __gmp_xsize, __gmp_yp, __gmp_ysize);
    return __gmp_c;
}
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_sub_1)
#if !defined(__GMP_FORCE_gpmpn_sub_1)
__GMP_EXTERN_INLINE
#endif
ANYCALLER static inline  mp_limb_t
gpmpn_sub_1(mp_ptr __gmp_dst, mp_srcptr __gmp_src, mp_size_t __gmp_size, mp_limb_t __gmp_n) __GMP_NOTHROW
{
    mp_limb_t __gmp_c;
    __GMPN_SUB_1(__gmp_c, __gmp_dst, __gmp_src, __gmp_size, __gmp_n);
    return __gmp_c;
}
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_neg)
#if !defined(__GMP_FORCE_gpmpn_neg)
__GMP_EXTERN_INLINE
#endif
ANYCALLER static inline mp_limb_t
gpmpn_neg(mp_ptr __gmp_rp, mp_srcptr __gmp_up, mp_size_t __gmp_n)
{
    while (*__gmp_up == 0) /* Low zero limbs are unchanged by negation. */
    {
        *__gmp_rp = 0;
        if (!--__gmp_n) /* All zero */
            return 0;
        ++__gmp_up;
        ++__gmp_rp;
    }

    *__gmp_rp = (-*__gmp_up) & GMP_NUMB_MASK;

    if (--__gmp_n) /* Higher limbs get complemented. */
        gpmpn_com(++__gmp_rp, ++__gmp_up, __gmp_n);

    return 1;
}
#endif

    }
}