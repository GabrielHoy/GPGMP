// Just includes all relevant MPN files
#pragma once
// General MPN functions used by GPGMP...
#include "PredictSizes.cuh"
#include "Allocation.cuh"
#include "Init.cuh"
#include "CopyArrayToArray.cuh"
#include "CopyToMPZArray.cuh"
#include "gpgmp.cuh"

// Mass forward declarations for GPGMP's versions of GMP functions

//TODO: This namespace is currently filled with mass copy-pasted function declarations from GMP's gmp-impl.h file - find which ones we don't already declare in GPGMP and remove them accordingly
/*
namespace gpgmp
{
    namespace mpnRoutines
    {


        __GPGMP_DECLSPEC void __gmpz_aorsmul_1 (REGPARM_3_1 (mpz_ptr, mpz_srcptr, mp_limb_t, mp_size_t)) REGPARM_ATTR(1);
        #define mpz_aorsmul_1(w,u,v,sub)  __gmpz_aorsmul_1 (REGPARM_3_1 (w, u, v, sub))

        #define mpz_n_pow_ui __gmpz_n_pow_ui
        __GPGMP_DECLSPEC void    mpz_n_pow_ui (mpz_ptr, mp_srcptr, mp_size_t, unsigned long);


        #define mpn_addmul_1c __GPGMP_MPN(addmul_1c)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

        #ifndef mpn_addmul_2  // if not done with cpuvec in a fat binary //
        #define mpn_addmul_2 __GPGMP_MPN(addmul_2)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_2 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
        #endif

        #define mpn_addmul_3 __GPGMP_MPN(addmul_3)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_3 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_addmul_4 __GPGMP_MPN(addmul_4)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_4 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_addmul_5 __GPGMP_MPN(addmul_5)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_5 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_addmul_6 __GPGMP_MPN(addmul_6)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_6 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_addmul_7 __GPGMP_MPN(addmul_7)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_7 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_addmul_8 __GPGMP_MPN(addmul_8)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_8 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        // Alternative entry point in mpn_addmul_2 for the benefit of mpn_sqr_basecase.  //
        #define mpn_addmul_2s __GPGMP_MPN(addmul_2s)
        __GPGMP_DECLSPEC mp_limb_t mpn_addmul_2s (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);












        #ifndef mpn_lshiftc  // if not done with cpuvec in a fat binary //
        #define mpn_lshiftc __GPGMP_MPN(lshiftc)
        __GPGMP_DECLSPEC mp_limb_t mpn_lshiftc (mp_ptr, mp_srcptr, mp_size_t, unsigned int);
        #endif

        #define mpn_add_err1_n  __GPGMP_MPN(add_err1_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_add_err1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_add_err2_n  __GPGMP_MPN(add_err2_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_add_err2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_add_err3_n  __GPGMP_MPN(add_err3_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_add_err3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_sub_err1_n  __GPGMP_MPN(sub_err1_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_sub_err1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_sub_err2_n  __GPGMP_MPN(sub_err2_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_sub_err2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_sub_err3_n  __GPGMP_MPN(sub_err3_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_sub_err3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_add_n_sub_n __GPGMP_MPN(add_n_sub_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_add_n_sub_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        #define mpn_add_n_sub_nc __GPGMP_MPN(add_n_sub_nc)
        __GPGMP_DECLSPEC mp_limb_t mpn_add_n_sub_nc (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_addaddmul_1msb0 __GPGMP_MPN(addaddmul_1msb0)
        __GPGMP_DECLSPEC mp_limb_t mpn_addaddmul_1msb0 (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

        #define mpn_divrem_1c __GPGMP_MPN(divrem_1c)
        __GPGMP_DECLSPEC mp_limb_t mpn_divrem_1c (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

        #define mpn_dump __GPGMP_MPN(dump)
        __GPGMP_DECLSPEC void mpn_dump (mp_srcptr, mp_size_t);

        #define mpn_fib2_ui __GPGMP_MPN(fib2_ui)
        __GPGMP_DECLSPEC mp_size_t mpn_fib2_ui (mp_ptr, mp_ptr, unsigned long);

        #define mpn_fib2m __GPGMP_MPN(fib2m)
        __GPGMP_DECLSPEC int mpn_fib2m (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

        #define mpn_strongfibo __GPGMP_MPN(strongfibo)
        __GPGMP_DECLSPEC int mpn_strongfibo (mp_srcptr, mp_size_t, mp_ptr);











        // Remap names of internal mpn functions.  //
        #define __clz_tab               __GPGMP_MPN(clz_tab)
        #define mpn_udiv_w_sdiv		__GPGMP_MPN(udiv_w_sdiv)

        #define mpn_jacobi_base __GPGMP_MPN(jacobi_base)
        __GPGMP_DECLSPEC int mpn_jacobi_base (mp_limb_t, mp_limb_t, int) ATTRIBUTE_CONST;

        #define mpn_jacobi_2 __GPGMP_MPN(jacobi_2)
        __GPGMP_DECLSPEC int mpn_jacobi_2 (mp_srcptr, mp_srcptr, unsigned);

        #define mpn_jacobi_n __GPGMP_MPN(jacobi_n)
        __GPGMP_DECLSPEC int mpn_jacobi_n (mp_ptr, mp_ptr, mp_size_t, unsigned);

        #define mpn_mod_1c __GPGMP_MPN(mod_1c)
        __GPGMP_DECLSPEC mp_limb_t mpn_mod_1c (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

        #define mpn_mul_1c __GPGMP_MPN(mul_1c)
        __GPGMP_DECLSPEC mp_limb_t mpn_mul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

        #define mpn_mul_2 __GPGMP_MPN(mul_2)
        __GPGMP_DECLSPEC mp_limb_t mpn_mul_2 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_mul_3 __GPGMP_MPN(mul_3)
        __GPGMP_DECLSPEC mp_limb_t mpn_mul_3 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_mul_4 __GPGMP_MPN(mul_4)
        __GPGMP_DECLSPEC mp_limb_t mpn_mul_4 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_mul_5 __GPGMP_MPN(mul_5)
        __GPGMP_DECLSPEC mp_limb_t mpn_mul_5 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #define mpn_mul_6 __GPGMP_MPN(mul_6)
        __GPGMP_DECLSPEC mp_limb_t mpn_mul_6 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        #ifndef mpn_mul_basecase  // if not done with cpuvec in a fat binary //
        #define mpn_mul_basecase __GPGMP_MPN(mul_basecase)
        __GPGMP_DECLSPEC void mpn_mul_basecase (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
        #endif

        #define mpn_mullo_n __GPGMP_MPN(mullo_n)
        __GPGMP_DECLSPEC void mpn_mullo_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        #ifndef mpn_mullo_basecase  // if not done with cpuvec in a fat binary //
        #define mpn_mullo_basecase __GPGMP_MPN(mullo_basecase)
        __GPGMP_DECLSPEC void mpn_mullo_basecase (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
        #endif

        #ifndef mpn_sqr_basecase  // if not done with cpuvec in a fat binary //
        #define mpn_sqr_basecase __GPGMP_MPN(sqr_basecase)
        __GPGMP_DECLSPEC void mpn_sqr_basecase (mp_ptr, mp_srcptr, mp_size_t);
        #endif

        #define mpn_sqrlo __GPGMP_MPN(sqrlo)
        __GPGMP_DECLSPEC void mpn_sqrlo (mp_ptr, mp_srcptr, mp_size_t);

        #define mpn_sqrlo_basecase __GPGMP_MPN(sqrlo_basecase)
        __GPGMP_DECLSPEC void mpn_sqrlo_basecase (mp_ptr, mp_srcptr, mp_size_t);

        #define mpn_mulmid_basecase __GPGMP_MPN(mulmid_basecase)
        __GPGMP_DECLSPEC void mpn_mulmid_basecase (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

        #define mpn_mulmid_n __GPGMP_MPN(mulmid_n)
        __GPGMP_DECLSPEC void mpn_mulmid_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        #define mpn_mulmid __GPGMP_MPN(mulmid)
        __GPGMP_DECLSPEC void mpn_mulmid (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

        #define mpn_submul_1c __GPGMP_MPN(submul_1c)
        __GPGMP_DECLSPEC mp_limb_t mpn_submul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

        #ifndef mpn_redc_1  // if not done with cpuvec in a fat binary //
        #define mpn_redc_1 __GPGMP_MPN(redc_1)
        __GPGMP_DECLSPEC mp_limb_t mpn_redc_1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
        #endif

        #ifndef mpn_redc_2  // if not done with cpuvec in a fat binary //
        #define mpn_redc_2 __GPGMP_MPN(redc_2)
        __GPGMP_DECLSPEC mp_limb_t mpn_redc_2 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
        #endif

        #define mpn_redc_n __GPGMP_MPN(redc_n)
        __GPGMP_DECLSPEC void mpn_redc_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);









        #ifndef mpn_mod_1_1p_cps  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1_1p_cps __GPGMP_MPN(mod_1_1p_cps)
        __GPGMP_DECLSPEC void mpn_mod_1_1p_cps (mp_limb_t [4], mp_limb_t);
        #endif
        #ifndef mpn_mod_1_1p  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1_1p __GPGMP_MPN(mod_1_1p)
        __GPGMP_DECLSPEC mp_limb_t mpn_mod_1_1p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [4]) __GMP_ATTRIBUTE_PURE;
        #endif

        #ifndef mpn_mod_1s_2p_cps  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1s_2p_cps __GPGMP_MPN(mod_1s_2p_cps)
        __GPGMP_DECLSPEC void mpn_mod_1s_2p_cps (mp_limb_t [5], mp_limb_t);
        #endif
        #ifndef mpn_mod_1s_2p  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1s_2p __GPGMP_MPN(mod_1s_2p)
        __GPGMP_DECLSPEC mp_limb_t mpn_mod_1s_2p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [5]) __GMP_ATTRIBUTE_PURE;
        #endif

        #ifndef mpn_mod_1s_3p_cps  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1s_3p_cps __GPGMP_MPN(mod_1s_3p_cps)
        __GPGMP_DECLSPEC void mpn_mod_1s_3p_cps (mp_limb_t [6], mp_limb_t);
        #endif
        #ifndef mpn_mod_1s_3p  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1s_3p __GPGMP_MPN(mod_1s_3p)
        __GPGMP_DECLSPEC mp_limb_t mpn_mod_1s_3p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [6]) __GMP_ATTRIBUTE_PURE;
        #endif

        #ifndef mpn_mod_1s_4p_cps  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1s_4p_cps __GPGMP_MPN(mod_1s_4p_cps)
        __GPGMP_DECLSPEC void mpn_mod_1s_4p_cps (mp_limb_t [7], mp_limb_t);
        #endif
        #ifndef mpn_mod_1s_4p  // if not done with cpuvec in a fat binary //
        #define mpn_mod_1s_4p __GPGMP_MPN(mod_1s_4p)
        __GPGMP_DECLSPEC mp_limb_t mpn_mod_1s_4p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [7]) __GMP_ATTRIBUTE_PURE;
        #endif

        #define mpn_bc_mulmod_bnm1 __GPGMP_MPN(bc_mulmod_bnm1)
        __GPGMP_DECLSPEC void mpn_bc_mulmod_bnm1 (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
        #define mpn_mulmod_bnm1 __GPGMP_MPN(mulmod_bnm1)
        __GPGMP_DECLSPEC void mpn_mulmod_bnm1 (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define mpn_mulmod_bnm1_next_size __GPGMP_MPN(mulmod_bnm1_next_size)
        __GPGMP_DECLSPEC mp_size_t mpn_mulmod_bnm1_next_size (mp_size_t) ATTRIBUTE_CONST;
























        #define   mpn_sqr_diagonal __GPGMP_MPN(sqr_diagonal)
        __GPGMP_DECLSPEC void      mpn_sqr_diagonal (mp_ptr, mp_srcptr, mp_size_t);

        #define mpn_sqr_diag_addlsh1 __GPGMP_MPN(sqr_diag_addlsh1)
        __GPGMP_DECLSPEC void      mpn_sqr_diag_addlsh1 (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        #define   mpn_toom_interpolate_5pts __GPGMP_MPN(toom_interpolate_5pts)
        __GPGMP_DECLSPEC void      mpn_toom_interpolate_5pts (mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_limb_t);

        enum toom6_flags {toom6_all_pos = 0, toom6_vm1_neg = 1, toom6_vm2_neg = 2};
        #define   mpn_toom_interpolate_6pts __GPGMP_MPN(toom_interpolate_6pts)
        __GPGMP_DECLSPEC void      mpn_toom_interpolate_6pts (mp_ptr, mp_size_t, enum toom6_flags, mp_ptr, mp_ptr, mp_ptr, mp_size_t);

        enum toom7_flags { toom7_w1_neg = 1, toom7_w3_neg = 2 };
        #define   mpn_toom_interpolate_7pts __GPGMP_MPN(toom_interpolate_7pts)
        __GPGMP_DECLSPEC void      mpn_toom_interpolate_7pts (mp_ptr, mp_size_t, enum toom7_flags, mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

        #define mpn_toom_interpolate_8pts __GPGMP_MPN(toom_interpolate_8pts)
        __GPGMP_DECLSPEC void      mpn_toom_interpolate_8pts (mp_ptr, mp_size_t, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

        #define mpn_toom_interpolate_12pts __GPGMP_MPN(toom_interpolate_12pts)
        __GPGMP_DECLSPEC void      mpn_toom_interpolate_12pts (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_ptr);

        #define mpn_toom_interpolate_16pts __GPGMP_MPN(toom_interpolate_16pts)
        __GPGMP_DECLSPEC void      mpn_toom_interpolate_16pts (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_ptr);

        #define   mpn_toom_couple_handling __GPGMP_MPN(toom_couple_handling)
        __GPGMP_DECLSPEC void mpn_toom_couple_handling (mp_ptr, mp_size_t, mp_ptr, int, mp_size_t, int, int);

        #define   mpn_toom_eval_dgr3_pm1 __GPGMP_MPN(toom_eval_dgr3_pm1)
        __GPGMP_DECLSPEC int mpn_toom_eval_dgr3_pm1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

        #define   mpn_toom_eval_dgr3_pm2 __GPGMP_MPN(toom_eval_dgr3_pm2)
        __GPGMP_DECLSPEC int mpn_toom_eval_dgr3_pm2 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

        #define   mpn_toom_eval_pm1 __GPGMP_MPN(toom_eval_pm1)
        __GPGMP_DECLSPEC int mpn_toom_eval_pm1 (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

        #define   mpn_toom_eval_pm2 __GPGMP_MPN(toom_eval_pm2)
        __GPGMP_DECLSPEC int mpn_toom_eval_pm2 (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

        #define   mpn_toom_eval_pm2exp __GPGMP_MPN(toom_eval_pm2exp)
        __GPGMP_DECLSPEC int mpn_toom_eval_pm2exp (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, unsigned, mp_ptr);

        #define   mpn_toom_eval_pm2rexp __GPGMP_MPN(toom_eval_pm2rexp)
        __GPGMP_DECLSPEC int mpn_toom_eval_pm2rexp (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, unsigned, mp_ptr);

        #define   mpn_toom22_mul __GPGMP_MPN(toom22_mul)
        __GPGMP_DECLSPEC void      mpn_toom22_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom32_mul __GPGMP_MPN(toom32_mul)
        __GPGMP_DECLSPEC void      mpn_toom32_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom42_mul __GPGMP_MPN(toom42_mul)
        __GPGMP_DECLSPEC void      mpn_toom42_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom52_mul __GPGMP_MPN(toom52_mul)
        __GPGMP_DECLSPEC void      mpn_toom52_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom62_mul __GPGMP_MPN(toom62_mul)
        __GPGMP_DECLSPEC void      mpn_toom62_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom2_sqr __GPGMP_MPN(toom2_sqr)
        __GPGMP_DECLSPEC void      mpn_toom2_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom33_mul __GPGMP_MPN(toom33_mul)
        __GPGMP_DECLSPEC void      mpn_toom33_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom43_mul __GPGMP_MPN(toom43_mul)
        __GPGMP_DECLSPEC void      mpn_toom43_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom53_mul __GPGMP_MPN(toom53_mul)
        __GPGMP_DECLSPEC void      mpn_toom53_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom54_mul __GPGMP_MPN(toom54_mul)
        __GPGMP_DECLSPEC void      mpn_toom54_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom63_mul __GPGMP_MPN(toom63_mul)
        __GPGMP_DECLSPEC void      mpn_toom63_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom3_sqr __GPGMP_MPN(toom3_sqr)
        __GPGMP_DECLSPEC void      mpn_toom3_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom44_mul __GPGMP_MPN(toom44_mul)
        __GPGMP_DECLSPEC void      mpn_toom44_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom4_sqr __GPGMP_MPN(toom4_sqr)
        __GPGMP_DECLSPEC void      mpn_toom4_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom6h_mul __GPGMP_MPN(toom6h_mul)
        __GPGMP_DECLSPEC void      mpn_toom6h_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom6_sqr __GPGMP_MPN(toom6_sqr)
        __GPGMP_DECLSPEC void      mpn_toom6_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom8h_mul __GPGMP_MPN(toom8h_mul)
        __GPGMP_DECLSPEC void      mpn_toom8h_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom8_sqr __GPGMP_MPN(toom8_sqr)
        __GPGMP_DECLSPEC void      mpn_toom8_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_toom42_mulmid __GPGMP_MPN(toom42_mulmid)
        __GPGMP_DECLSPEC void      mpn_toom42_mulmid (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_fft_best_k __GPGMP_MPN(fft_best_k)
        __GPGMP_DECLSPEC int       mpn_fft_best_k (mp_size_t, int) ATTRIBUTE_CONST;

        #define   mpn_mul_fft __GPGMP_MPN(mul_fft)
        __GPGMP_DECLSPEC mp_limb_t mpn_mul_fft (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, int);

        #define   mpn_mul_fft_full __GPGMP_MPN(mul_fft_full)
        __GPGMP_DECLSPEC void      mpn_mul_fft_full (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

        #define   mpn_nussbaumer_mul __GPGMP_MPN(nussbaumer_mul)
        __GPGMP_DECLSPEC void      mpn_nussbaumer_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

        #define   mpn_fft_next_size __GPGMP_MPN(fft_next_size)
        __GPGMP_DECLSPEC mp_size_t mpn_fft_next_size (mp_size_t, int) ATTRIBUTE_CONST;

        #define   mpn_div_qr_1n_pi1 __GPGMP_MPN(div_qr_1n_pi1)
        __GPGMP_DECLSPEC mp_limb_t mpn_div_qr_1n_pi1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

        #define   mpn_div_qr_2n_pi1 __GPGMP_MPN(div_qr_2n_pi1)
        __GPGMP_DECLSPEC mp_limb_t mpn_div_qr_2n_pi1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

        #define   mpn_div_qr_2u_pi1 __GPGMP_MPN(div_qr_2u_pi1)
        __GPGMP_DECLSPEC mp_limb_t mpn_div_qr_2u_pi1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int, mp_limb_t);

        #define   mpn_sbpi1_div_qr __GPGMP_MPN(sbpi1_div_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_sbpi1_div_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        #define   mpn_sbpi1_div_q __GPGMP_MPN(sbpi1_div_q)
        __GPGMP_DECLSPEC mp_limb_t mpn_sbpi1_div_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        #define   mpn_sbpi1_divappr_q __GPGMP_MPN(sbpi1_divappr_q)
        __GPGMP_DECLSPEC mp_limb_t mpn_sbpi1_divappr_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        #define   mpn_dcpi1_div_qr __GPGMP_MPN(dcpi1_div_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_dcpi1_div_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);
        #define   mpn_dcpi1_div_qr_n __GPGMP_MPN(dcpi1_div_qr_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_dcpi1_div_qr_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, gmp_pi1_t *, mp_ptr);

        #define   mpn_dcpi1_div_q __GPGMP_MPN(dcpi1_div_q)
        __GPGMP_DECLSPEC mp_limb_t mpn_dcpi1_div_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);

        #define   mpn_dcpi1_divappr_q __GPGMP_MPN(dcpi1_divappr_q)
        __GPGMP_DECLSPEC mp_limb_t mpn_dcpi1_divappr_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);

        #define   mpn_mu_div_qr __GPGMP_MPN(mu_div_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_mu_div_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_mu_div_qr_itch __GPGMP_MPN(mu_div_qr_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_mu_div_qr_itch (mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;

        #define   mpn_preinv_mu_div_qr __GPGMP_MPN(preinv_mu_div_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_preinv_mu_div_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_preinv_mu_div_qr_itch __GPGMP_MPN(preinv_mu_div_qr_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_preinv_mu_div_qr_itch (mp_size_t, mp_size_t, mp_size_t) ATTRIBUTE_CONST;

        #define   mpn_mu_divappr_q __GPGMP_MPN(mu_divappr_q)
        __GPGMP_DECLSPEC mp_limb_t mpn_mu_divappr_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_mu_divappr_q_itch __GPGMP_MPN(mu_divappr_q_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_mu_divappr_q_itch (mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;

        #define   mpn_mu_div_q __GPGMP_MPN(mu_div_q)
        __GPGMP_DECLSPEC mp_limb_t mpn_mu_div_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_mu_div_q_itch __GPGMP_MPN(mu_div_q_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_mu_div_q_itch (mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;

        #define  mpn_div_q __GPGMP_MPN(div_q)
        __GPGMP_DECLSPEC void mpn_div_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

        #define   mpn_invert __GPGMP_MPN(invert)
        __GPGMP_DECLSPEC void      mpn_invert (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
        #define mpn_invert_itch(n)  mpn_invertappr_itch(n)

        #define   mpn_ni_invertappr __GPGMP_MPN(ni_invertappr)
        __GPGMP_DECLSPEC mp_limb_t mpn_ni_invertappr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_invertappr __GPGMP_MPN(invertappr)
        __GPGMP_DECLSPEC mp_limb_t mpn_invertappr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
        #define mpn_invertappr_itch(n)  (2 * (n))

        #define   mpn_binvert __GPGMP_MPN(binvert)
        __GPGMP_DECLSPEC void      mpn_binvert (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_binvert_itch __GPGMP_MPN(binvert_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_binvert_itch (mp_size_t) ATTRIBUTE_CONST;

        #define mpn_bdiv_q_1 __GPGMP_MPN(bdiv_q_1)
        __GPGMP_DECLSPEC mp_limb_t mpn_bdiv_q_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_pi1_bdiv_q_1 __GPGMP_MPN(pi1_bdiv_q_1)
        __GPGMP_DECLSPEC mp_limb_t mpn_pi1_bdiv_q_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int);

        #define   mpn_sbpi1_bdiv_qr __GPGMP_MPN(sbpi1_bdiv_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_sbpi1_bdiv_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        #define   mpn_sbpi1_bdiv_q __GPGMP_MPN(sbpi1_bdiv_q)
        __GPGMP_DECLSPEC void      mpn_sbpi1_bdiv_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        #define   mpn_sbpi1_bdiv_r __GPGMP_MPN(sbpi1_bdiv_r)
        __GPGMP_DECLSPEC mp_limb_t mpn_sbpi1_bdiv_r (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        #define   mpn_dcpi1_bdiv_qr __GPGMP_MPN(dcpi1_bdiv_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_dcpi1_bdiv_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
        #define   mpn_dcpi1_bdiv_qr_n_itch __GPGMP_MPN(dcpi1_bdiv_qr_n_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_dcpi1_bdiv_qr_n_itch (mp_size_t) ATTRIBUTE_CONST;

        #define   mpn_dcpi1_bdiv_qr_n __GPGMP_MPN(dcpi1_bdiv_qr_n)
        __GPGMP_DECLSPEC mp_limb_t mpn_dcpi1_bdiv_qr_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
        #define   mpn_dcpi1_bdiv_q __GPGMP_MPN(dcpi1_bdiv_q)
        __GPGMP_DECLSPEC void      mpn_dcpi1_bdiv_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        #define   mpn_mu_bdiv_qr __GPGMP_MPN(mu_bdiv_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_mu_bdiv_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_mu_bdiv_qr_itch __GPGMP_MPN(mu_bdiv_qr_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_mu_bdiv_qr_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

        #define   mpn_mu_bdiv_q __GPGMP_MPN(mu_bdiv_q)
        __GPGMP_DECLSPEC void      mpn_mu_bdiv_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_mu_bdiv_q_itch __GPGMP_MPN(mu_bdiv_q_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_mu_bdiv_q_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

        #define   mpn_bdiv_qr __GPGMP_MPN(bdiv_qr)
        __GPGMP_DECLSPEC mp_limb_t mpn_bdiv_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_bdiv_qr_itch __GPGMP_MPN(bdiv_qr_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_bdiv_qr_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

        #define   mpn_bdiv_q __GPGMP_MPN(bdiv_q)
        __GPGMP_DECLSPEC void      mpn_bdiv_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_bdiv_q_itch __GPGMP_MPN(bdiv_q_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_bdiv_q_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

        #define   mpn_divexact __GPGMP_MPN(divexact)
        __GPGMP_DECLSPEC void      mpn_divexact (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
        #define   mpn_divexact_itch __GPGMP_MPN(divexact_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_divexact_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

        #ifndef mpn_bdiv_dbm1c  // if not done with cpuvec in a fat binary //
        #define   mpn_bdiv_dbm1c __GPGMP_MPN(bdiv_dbm1c)
        __GPGMP_DECLSPEC mp_limb_t mpn_bdiv_dbm1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
        #endif

        #define   mpn_bdiv_dbm1(dst, src, size, divisor) \
        mpn_bdiv_dbm1c (dst, src, size, divisor, __GMP_CAST (mp_limb_t, 0))

        #define   mpn_powm __GPGMP_MPN(powm)
        __GPGMP_DECLSPEC void      mpn_powm (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_powlo __GPGMP_MPN(powlo)
        __GPGMP_DECLSPEC void      mpn_powlo (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

        #define mpn_sec_pi1_div_qr __GPGMP_MPN(sec_pi1_div_qr)
        #define mpn_sec_pi1_div_r __GPGMP_MPN(sec_pi1_div_r)
        __GPGMP_DECLSPEC mp_limb_t mpn_sec_pi1_div_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
        #define mpn_sec_pi1_div_r __GPGMP_MPN(sec_pi1_div_r)
        __GPGMP_DECLSPEC void mpn_sec_pi1_div_r (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);




















        #ifndef DIVEXACT_BY3_METHOD
        #if GMP_NUMB_BITS % 2 == 0 && ! defined (HAVE_NATIVE_mpn_divexact_by3c)
        #define DIVEXACT_BY3_METHOD 0	// default to using mpn_bdiv_dbm1c
        #else
        #define DIVEXACT_BY3_METHOD 1
        #endif
        #endif

        #if DIVEXACT_BY3_METHOD == 0
        #undef mpn_divexact_by3
        #define mpn_divexact_by3(dst,src,size) \
        (3 & mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 3)))
        // override mpn_divexact_by3c defined in gmp.h

        //#undef mpn_divexact_by3c
        //#define mpn_divexact_by3c(dst,src,size,cy) \
        //(3 & mpn_bdiv_dbm1c (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 3, GMP_NUMB_MASK / 3 * cy)))

        #endif

        #if GMP_NUMB_BITS % 4 == 0
        #define mpn_divexact_by5(dst,src,size) \
        (7 & 3 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 5)))
        #endif

        #if GMP_NUMB_BITS % 3 == 0
        #define mpn_divexact_by7(dst,src,size) \
        (7 & 1 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 7)))
        #endif

        #if GMP_NUMB_BITS % 6 == 0
        #define mpn_divexact_by9(dst,src,size) \
        (15 & 7 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 9)))
        #endif

        #if GMP_NUMB_BITS % 10 == 0
        #define mpn_divexact_by11(dst,src,size) \
        (15 & 5 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 11)))
        #endif

        #if GMP_NUMB_BITS % 12 == 0
        #define mpn_divexact_by13(dst,src,size) \
        (15 & 3 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 13)))
        #endif

        #if GMP_NUMB_BITS % 4 == 0
        #define mpn_divexact_by15(dst,src,size) \
        (15 & 1 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 15)))
        #endif

        #if GMP_NUMB_BITS % 8 == 0
        #define mpn_divexact_by17(dst,src,size) \
        (31 & 15 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 17)))
        #endif








        //#define mpz_divexact_gcd  __gmpz_divexact_gcd
        //__GPGMP_DECLSPEC void    mpz_divexact_gcd (mpz_ptr, mpz_srcptr, mpz_srcptr);
//
        //#define mpz_prodlimbs  __gmpz_prodlimbs
        //__GPGMP_DECLSPEC mp_size_t mpz_prodlimbs (mpz_ptr, mp_ptr, mp_size_t);
//
        //#define mpz_oddfac_1  __gmpz_oddfac_1
        //__GPGMP_DECLSPEC void mpz_oddfac_1 (mpz_ptr, mp_limb_t, unsigned);
//
        //#define mpz_stronglucas  __gmpz_stronglucas
        //__GPGMP_DECLSPEC int mpz_stronglucas (mpz_srcptr, mpz_ptr, mpz_ptr);
//
        //#define mpz_lucas_mod  __gmpz_lucas_mod
        //__GPGMP_DECLSPEC int mpz_lucas_mod (mpz_ptr, mpz_ptr, long, mp_bitcnt_t, mpz_srcptr, mpz_ptr, mpz_ptr);
//
        //#define mpz_inp_str_nowhite __gmpz_inp_str_nowhite
        //#ifdef _GMP_H_HAVE_FILE
        //__GPGMP_DECLSPEC size_t  mpz_inp_str_nowhite (mpz_ptr, FILE *, int, int, size_t);
        //#endif

        #define mpn_divisible_p __GPGMP_MPN(divisible_p)
        __GPGMP_DECLSPEC int     mpn_divisible_p (mp_srcptr, mp_size_t, mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;

        #define   mpn_rootrem __GPGMP_MPN(rootrem)
        __GPGMP_DECLSPEC mp_size_t mpn_rootrem (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_broot __GPGMP_MPN(broot)
        __GPGMP_DECLSPEC void mpn_broot (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_broot_invm1 __GPGMP_MPN(broot_invm1)
        __GPGMP_DECLSPEC void mpn_broot_invm1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        #define mpn_brootinv __GPGMP_MPN(brootinv)
        __GPGMP_DECLSPEC void mpn_brootinv (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);

        #define mpn_bsqrt __GPGMP_MPN(bsqrt)
        __GPGMP_DECLSPEC void mpn_bsqrt (mp_ptr, mp_srcptr, mp_bitcnt_t, mp_ptr);

        #define mpn_bsqrtinv __GPGMP_MPN(bsqrtinv)
        __GPGMP_DECLSPEC int mpn_bsqrtinv (mp_ptr, mp_srcptr, mp_bitcnt_t, mp_ptr);







        #ifndef mpn_modexact_1c_odd  // if not done with cpuvec in a fat binary //
        #define mpn_modexact_1c_odd __GPGMP_MPN(modexact_1c_odd)
        __GPGMP_DECLSPEC mp_limb_t mpn_modexact_1c_odd (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;
        #endif

        #if HAVE_NATIVE_mpn_modexact_1_odd
        #define   mpn_modexact_1_odd  __GPGMP_MPN(modexact_1_odd)
        __GPGMP_DECLSPEC mp_limb_t mpn_modexact_1_odd (mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;
        #else
        #define mpn_modexact_1_odd(src,size,divisor) \
        mpn_modexact_1c_odd (src, size, divisor, CNST_LIMB(0))
        #endif










        #define mpn_hgcd_matrix_init __GPGMP_MPN (hgcd_matrix_init)
        __GPGMP_DECLSPEC void mpn_hgcd_matrix_init (struct hgcd_matrix *, mp_size_t, mp_ptr);

        #define mpn_hgcd_matrix_update_q __GPGMP_MPN (hgcd_matrix_update_q)
        __GPGMP_DECLSPEC void mpn_hgcd_matrix_update_q (struct hgcd_matrix *, mp_srcptr, mp_size_t, unsigned, mp_ptr);

        #define mpn_hgcd_matrix_mul_1 __GPGMP_MPN (hgcd_matrix_mul_1)
        __GPGMP_DECLSPEC void mpn_hgcd_matrix_mul_1 (struct hgcd_matrix *, const struct hgcd_matrix1 *, mp_ptr);

        #define mpn_hgcd_matrix_mul __GPGMP_MPN (hgcd_matrix_mul)
        __GPGMP_DECLSPEC void mpn_hgcd_matrix_mul (struct hgcd_matrix *, const struct hgcd_matrix *, mp_ptr);

        #define mpn_hgcd_matrix_adjust __GPGMP_MPN (hgcd_matrix_adjust)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_matrix_adjust (const struct hgcd_matrix *, mp_size_t, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

        #define mpn_hgcd_step __GPGMP_MPN(hgcd_step)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_step (mp_size_t, mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

        #define mpn_hgcd_reduce __GPGMP_MPN(hgcd_reduce)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_reduce (struct hgcd_matrix *, mp_ptr, mp_ptr, mp_size_t, mp_size_t, mp_ptr);

        #define mpn_hgcd_reduce_itch __GPGMP_MPN(hgcd_reduce_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_reduce_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

        #define mpn_hgcd_itch __GPGMP_MPN (hgcd_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_itch (mp_size_t) ATTRIBUTE_CONST;

        #define mpn_hgcd __GPGMP_MPN (hgcd)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

        #define mpn_hgcd_appr_itch __GPGMP_MPN (hgcd_appr_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_appr_itch (mp_size_t) ATTRIBUTE_CONST;

        #define mpn_hgcd_appr __GPGMP_MPN (hgcd_appr)
        __GPGMP_DECLSPEC int mpn_hgcd_appr (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

        #define mpn_hgcd_jacobi __GPGMP_MPN (hgcd_jacobi)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_jacobi (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, unsigned *, mp_ptr);





        #define mpn_hgcd2 __GPGMP_MPN (hgcd2)
        __GPGMP_DECLSPEC int mpn_hgcd2 (mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t,	struct hgcd_matrix1 *);

        #define mpn_hgcd_mul_matrix1_vector __GPGMP_MPN (hgcd_mul_matrix1_vector)
        __GPGMP_DECLSPEC mp_size_t mpn_hgcd_mul_matrix1_vector (const struct hgcd_matrix1 *, mp_ptr, mp_srcptr, mp_ptr, mp_size_t);

        #define mpn_matrix22_mul1_inverse_vector __GPGMP_MPN (matrix22_mul1_inverse_vector)
        __GPGMP_DECLSPEC mp_size_t mpn_matrix22_mul1_inverse_vector (const struct hgcd_matrix1 *, mp_ptr, mp_srcptr, mp_ptr, mp_size_t);

        #define mpn_hgcd2_jacobi __GPGMP_MPN (hgcd2_jacobi)
        __GPGMP_DECLSPEC int mpn_hgcd2_jacobi (mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t, struct hgcd_matrix1 *, unsigned *);




        #define mpn_gcd_subdiv_step __GPGMP_MPN(gcd_subdiv_step)
        __GPGMP_DECLSPEC mp_size_t mpn_gcd_subdiv_step (mp_ptr, mp_ptr, mp_size_t, mp_size_t, gcd_subdiv_step_hook *, void *, mp_ptr);



        #define   mpn_matrix22_mul __GPGMP_MPN(matrix22_mul)
        __GPGMP_DECLSPEC void      mpn_matrix22_mul (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
        #define   mpn_matrix22_mul_itch __GPGMP_MPN(matrix22_mul_itch)
        __GPGMP_DECLSPEC mp_size_t mpn_matrix22_mul_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;


        #define mpn_sqrmod_bknp1 __GPGMP_MPN(sqrmod_bknp1)
        __GPGMP_DECLSPEC void mpn_sqrmod_bknp1 (mp_ptr, mp_srcptr, mp_size_t, unsigned, mp_ptr);
        #define mpn_mulmod_bknp1 __GPGMP_MPN(mulmod_bknp1)
        __GPGMP_DECLSPEC void mpn_mulmod_bknp1 (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned, mp_ptr);



        #ifndef mpn_mod_34lsub1  // if not done with cpuvec in a fat binary *
        #define mpn_mod_34lsub1 __GPGMP_MPN(mod_34lsub1)
        __GPGMP_DECLSPEC mp_limb_t mpn_mod_34lsub1 (mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;
        #endif
    }
}

*/



//"MPN" routines used by GPGMP.
// Most-to-all of these are ported over from the GMP library, and possibly rewritten.
// https://gmplib.org/manual/Low_002dlevel-Functions
/*
refactor note dump...
    - find GMP_NUMB_HIGHBIT
    - find BMOD_1_TO_MOD_1_THRESHOLD
    - find ASSERT_NOCARRY
    - find LOW_ZEROS_MASK
    - find MPN_CMP
    - find MPN_DECR_U subroutine define?
    - find MPN_DIVREM_OR_DIVEXACT_1 subroutine define?
    - find MPN_MOD_OR_MODEXACT_1_ODD subroutine define?
    - find MPN_FIB2_SIZE
    - find FIB_TABLE_LIMIT
    - find FIB_TABLE
    - find MP_LIMB_T_SWAP
    - find MP_PTR_SWAP
    - find GCDEXT_DC_THRESHOLD
    - find ASSERT_MPN_NONZERO_P
    - find LIMB_HIGHBIT_TO_MASK
    - find MPN_DECR_U
    - mp_double_limb_t is undefined atm
    - gcd_subdiv_step_hook is undefined atm
    - find MPN_GCD_SUBDIV_STEP_ITCH
    - find GCD_DC_THRESHOLD
    - find MPN_HGCD_MATRIX_INIT_ITCH

    (maybe defined later on in the mpn files that i havent gotten to yet?)
    - find mpn_hgcd_itch
    - find mpn_hgcd_matrix_init
    - find mpn_hgcd
    - find mpn_hgcd_matrix_adjust
    - find mpn_hgcd2
    - find mpn_matrix22_mul1_inverse_vector
    - find mpn_hgcd_mul_matrix1_vector
    (end of that portion)

    - find MPN_EXTRACT_NUMB
    - find NEG_CAST
    - find MP_BASE_AS_DOUBLE
    - find HGCD_THRESHOLD
    - find HGCD_APPR_THRESHOLD
    -



    possibly find a way to enable ""temp allocations"" in CUDA? memory pools? static/constants?
*/

/*
#include "MPNRoutines/add_1.cuh"
#include "MPNRoutines/add_err1_n.cuh"
#include "MPNRoutines/add_err2_n.cuh"
#include "MPNRoutines/add_err3_n.cuh"
#include "MPNRoutines/add_n_sub_n.cuh" //Warp Divergence
#include "MPNRoutines/add_n.cuh"
#include "MPNRoutines/add.cuh"
#include "MPNRoutines/com.cuh"
#include "MPNRoutines/comb_tables.cuh"
#include "MPNRoutines/compute_powtab.cuh"
#include "MPNRoutines/copyd.cuh"
#include "MPNRoutines/copyi.cuh"
#include "MPNRoutines/zero_p.cuh"
#include "MPNRoutines/zero.cuh"
#include "MPNRoutines/cnd_swap.cuh"
#include "MPNRoutines/cmp.cuh"
#include "MPNRoutines/cnd_add_n.cuh"
#include "MPNRoutines/neg.cuh"
#include "MPNRoutines/sub.cuh"
#include "MPNRoutines/sub_n.cuh"
#include "MPNRoutines/sub_1.cuh"
#include "MPNRoutines/sub_err1_n.cuh"
#include "MPNRoutines/sub_err2_n.cuh"
#include "MPNRoutines/sub_err3_n.cuh"
#include "MPNRoutines/cnd_sub_n.cuh"
#include "MPNRoutines/addmul_1.cuh"
#include "MPNRoutines/rshift.cuh"
#include "MPNRoutines/lshift.cuh"
#include "MPNRoutines/lshiftc.cuh"
#include "MPNRoutines/dump.cuh"
#include "MPNRoutines/scan0.cuh"
#include "MPNRoutines/bdiv_dbm1c.cuh"
#include "MPNRoutines/bdiv_q_1.cuh"
#include "MPNRoutines/bdiv_q.cuh" //Warp Divergence
#include "MPNRoutines/bdiv_qr.cuh" //Warp Divergence
#include "MPNRoutines/binvert.cuh"
#include "MPNRoutines/broot.cuh" //Possible Warp Divergence, Temporary Allocation
#include "MPNRoutines/brootinv.cuh" //Possible Warp Divergence
#include "MPNRoutines/bsqrt.cuh" //Lots of Sub-Routines
#include "MPNRoutines/bsqrtinv.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/dcpi1_bdiv_q.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/dcpi1_bdiv_qr.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/dcpi1_div_q.cuh" //Sub-Routine, Warp Divergence, Temporary Allocation, Unknown Constant
#include "MPNRoutines/dcpi1_div_qr.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/dcpi1_divappr_q.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/div_q.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/div_qr_1.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/div_qr_1n_pi1.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/div_qr_1n_pi2.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/div_qr_1u_pi2.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/div_qr_2.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/div_qr_2n_pi1.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/div_qr_2u_pi1.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/dive_1.cuh" //Warp Divergence
#include "MPNRoutines/diveby3.cuh"
#include "MPNRoutines/divexact.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/divis.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/divrem_1.cuh" //Warp Divergence
#include "MPNRoutines/divrem_2.cuh" //Warp Divergence
#include "MPNRoutines/divrem.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/fib2_ui.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/fib2m.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/gcd_1.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/gcd_11.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/gcd_22.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/gcd_subdiv_step.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/gcd.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/gcdext_1.cuh" //Warp Divergence
#include "MPNRoutines/gcdext_lehmer.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/gcdext.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation
#include "MPNRoutines/get_d.cuh" //Lots of Sub-Routines, Warp Divergence, Temporary Allocation, Unknown Include
#include "MPNRoutines/hgcd_appr.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/hgcd_jacobi.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/hgcd_matrix.cuh" //Lots of Sub-Routines, Warp Divergence
#include "MPNRoutines/hgcd_reduce.cuh" //Lots of Sub-Routines, Warp Divergence

//.....from here on i didnt bother attaching ANYCALLER's, indenting, fixing undefined macros, etc etc etc - need to come back and do so later
//All of these are probably bad in some way, should be assumed
#include "MPNRoutines/hgcd_step.cuh"
#include "MPNRoutines/hgcd.cuh"
#include "MPNRoutines/hgcd2_jacobi.cuh"
#include "MPNRoutines/hgcd2-div.cuh"
#include "MPNRoutines/hgcd2.cuh"
#include "MPNRoutines/invertappr.cuh"
#include "MPNRoutines/invert.cuh"
#include "MPNRoutines/jacbase.cuh"
#include "MPNRoutines/jacobi_2.cuh"
#include "MPNRoutines/jacobi.cuh"
#include "MPNRoutines/logops_n.cuh"
#include "MPNRoutines/matrix22_mul.cuh"
#include "MPNRoutines/matrix22_mul1_inverse_vector.cuh"
#include "MPNRoutines/mod_1_1.cuh"
#include "MPNRoutines/mod_1_2.cuh"
#include "MPNRoutines/mod_1_3.cuh"
#include "MPNRoutines/mod_1_4.cuh"
#include "MPNRoutines/mod_1.cuh"
#include "MPNRoutines/mod_34lsub1.cuh"
#include "MPNRoutines/mode1o.cuh"
#include "MPNRoutines/mu_bdiv_q.cuh"
#include "MPNRoutines/mu_bdiv_qr.cuh"
#include "MPNRoutines/mu_div_q.cuh"
#include "MPNRoutines/mu_div_qr.cuh"
#include "MPNRoutines/mu_divappr_q.cuh"
#include "MPNRoutines/mul_1.cuh"
#include "MPNRoutines/mul_basecase.cuh"
#include "MPNRoutines/mul_fft.cuh"
#include "MPNRoutines/mul_n.cuh"
#include "MPNRoutines/mul.cuh"
#include "MPNRoutines/mullo_basecase.cuh"
#include "MPNRoutines/mullo_n.cuh"
#include "MPNRoutines/mulmid_basecase.cuh"
#include "MPNRoutines/mulmid_n.cuh"
#include "MPNRoutines/mulmid.cuh"
#include "MPNRoutines/mulmod_bknp1.cuh"
#include "MPNRoutines/mulmod_bnm1.cuh"
#include "MPNRoutines/nussbaumer_mul.cuh"
#include "MPNRoutines/trialdiv.cuh"
#include "MPNRoutines/perfpow.cuh"
#include "MPNRoutines/perfsqr.cuh"
#include "MPNRoutines/popham.cuh"
#include "MPNRoutines/pow_1.cuh"
#include "MPNRoutines/powlo.cuh"
#include "MPNRoutines/powm.cuh"
#include "MPNRoutines/pre_divrem_1.cuh"
#include "MPNRoutines/pre_mod_1.cuh"
//#include "MPNRoutines/random.cuh" //soooo much external dependent defines and variables and headache...come back to later...
//#include "MPNRoutines/random2.cuh" //soooo much external dependent defines and variables and headache...come back to later...
#include "MPNRoutines/redc_1.cuh"
#include "MPNRoutines/redc_2.cuh" //THIS IS WHERE I LEFT OFF ON 06/04/2025 ADDING ANYCALLERS
#include "MPNRoutines/redc_n.cuh"
#include "MPNRoutines/remove.cuh"
#include "MPNRoutines/rootrem.cuh"
#include "MPNRoutines/sbpi1_bdiv_q.cuh"
#include "MPNRoutines/sbpi1_bdiv_qr.cuh"
#include "MPNRoutines/sbpi1_bdiv_r.cuh"
#include "MPNRoutines/sbpi1_div_q.cuh"
#include "MPNRoutines/sbpi1_div_qr.cuh"
#include "MPNRoutines/sbpi1_divappr_q.cuh"
#include "MPNRoutines/sec_aors_1.cuh"
#include "MPNRoutines/sec_div.cuh"
#include "MPNRoutines/sec_invert.cuh"
#include "MPNRoutines/sec_mul.cuh"
#include "MPNRoutines/sec_pi1_div.cuh" //Multi-Function File
#include "MPNRoutines/sec_powm.cuh"
#include "MPNRoutines/sec_sqr.cuh"
#include "MPNRoutines/sec_tabselect.cuh"
#include "MPNRoutines/set_str.cuh"
#include "MPNRoutines/sizeinbase.cuh"
#include "MPNRoutines/sqr_basecase.cuh"
#include "MPNRoutines/sqr.cuh"
#include "MPNRoutines/sqrlo_basecase.cuh"
#include "MPNRoutines/sqrlo.cuh"
#include "MPNRoutines/sqrmod_bnm1.cuh"
#include "MPNRoutines/sqrtrem.cuh"
#include "MPNRoutines/strongfibo.cuh"
#include "MPNRoutines/sub_1.cuh"
#include "MPNRoutines/sub_err1_n.cuh"
#include "MPNRoutines/sub_err2_n.cuh"
#include "MPNRoutines/sub_err3_n.cuh"
#include "MPNRoutines/sub_n.cuh"
#include "MPNRoutines/sub.cuh"
#include "MPNRoutines/submul_1.cuh"
#include "MPNRoutines/tdiv_qr.cuh"
#include "MPNRoutines/toom_couple_handling.cuh"
#include "MPNRoutines/toom_eval_dgr3_pm1.cuh"
#include "MPNRoutines/toom_eval_dgr3_pm2.cuh"
#include "MPNRoutines/toom_eval_pm1.cuh"
#include "MPNRoutines/toom_eval_pm2.cuh"
#include "MPNRoutines/toom_eval_pm2exp.cuh"
#include "MPNRoutines/toom_eval_pm2rexp.cuh"
#include "MPNRoutines/toom_interpolate_5pts.cuh"
#include "MPNRoutines/toom_interpolate_6pts.cuh"
#include "MPNRoutines/toom_interpolate_7pts.cuh"
#include "MPNRoutines/toom_interpolate_8pts.cuh"
#include "MPNRoutines/toom_interpolate_12pts.cuh"
#include "MPNRoutines/toom_interpolate_16pts.cuh"
#include "MPNRoutines/toom2_sqr.cuh"
#include "MPNRoutines/toom3_sqr.cuh"
#include "MPNRoutines/toom4_sqr.cuh"
#include "MPNRoutines/toom6_sqr.cuh"
#include "MPNRoutines/toom6h_mul.cuh"
#include "MPNRoutines/toom8_sqr.cuh"
#include "MPNRoutines/toom8h_mul.cuh"
#include "MPNRoutines/toom22_mul.cuh"
#include "MPNRoutines/toom32_mul.cuh"
#include "MPNRoutines/toom33_mul.cuh"
#include "MPNRoutines/toom42_mul.cuh"
#include "MPNRoutines/toom42_mulmid.cuh"
#include "MPNRoutines/toom43_mul.cuh"
#include "MPNRoutines/toom44_mul.cuh"
#include "MPNRoutines/toom52_mul.cuh"
#include "MPNRoutines/toom53_mul.cuh"
#include "MPNRoutines/toom54_mul.cuh"
#include "MPNRoutines/toom62_mul.cuh"
#include "MPNRoutines/toom63_mul.cuh"
//#include "MPNRoutines/udiv_w_sdiv.cuh" seems like the function sdiv_qrnnd in this file is only included in libgcc and the functions declared in here are never used in GMP - not sure if its even worth it to approach porting








*/

//#include "gmp-mparam.h" //Seemingly useless
