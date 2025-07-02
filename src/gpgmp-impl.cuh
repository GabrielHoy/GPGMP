// This file is NOT equivalent to gmp-impl.h, it has been significantly modified to fit GPGMP.

/* Include file for internal GNU MP types and definitions.

   THE CONTENTS OF THIS FILE ARE FOR INTERNAL USE AND ARE ALMOST CERTAIN TO
   BE SUBJECT TO INCOMPATIBLE CHANGES IN FUTURE GNU MP RELEASES.

Copyright 1991-2018, 2021, 2022 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

/* __GPGMP_DECLSPEC __GPGMP_CALLERTYPE must be given on any global data that will be accessed
   from outside libgmp, meaning from the test or development programs, or
   from libgmpxx.  Failing to do this will result in an incorrect address
   being used for the accesses.  On functions __GPGMP_DECLSPEC __GPGMP_CALLERTYPE makes calls
   from outside libgmp more efficient, but they'll still work fine without
   it.  */
#pragma once

#ifndef __GPGMP_IMPL_H__
#define __GPGMP_IMPL_H__

#if defined _CRAY
#include <intrinsics.h> /* for _popcnt */
#endif

/* For INT_MAX, etc. We used to avoid it because of a bug (on solaris,
   gcc 2.95 under -mcpu=ultrasparc in ABI=32 ends up getting wrong
   values (the ABI=64 values)), but it should be safe now.

   On Cray vector systems, however, we need the system limits.h since sizes
   of signed and unsigned types can differ there, depending on compiler
   options (eg. -hnofastmd), making our SHRT_MAX etc expressions fail.  For
   reference, int can be 46 or 64 bits, whereas uint is always 64 bits; and
   short can be 24, 32, 46 or 64 bits, and different for ushort.  */

#include <limits.h>

/* For fat.h and other fat binary stuff.
   No need for __GMP_ATTRIBUTE_PURE or __GMP_NOTHROW, since functions
   declared this way are only used to set function pointers in __ggpmpn_cpuvec,
   they're not called directly.  */
#define DECL_add_n(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_addlsh1_n(name) \
    DECL_add_n(name)
#define DECL_addlsh2_n(name) \
    DECL_add_n(name)
#define DECL_addmul_1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_addmul_2(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr)
#define DECL_bdiv_dbm1c(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t)
#define DECL_cnd_add_n(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_cnd_sub_n(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_com(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void name(mp_ptr, mp_srcptr, mp_size_t)
#define DECL_copyd(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void name(mp_ptr, mp_srcptr, mp_size_t)
#define DECL_copyi(name) \
    DECL_copyd(name)
#define DECL_divexact_1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void name(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_divexact_by3c(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_divrem_1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_gcd_11(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_limb_t, mp_limb_t)
#define DECL_lshift(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_srcptr, mp_size_t, unsigned)
#define DECL_lshiftc(name) \
    DECL_lshift(name)
#define DECL_mod_1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_mod_1_1p(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t[])
#define DECL_mod_1_1p_cps(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void name(mp_limb_t cps[], mp_limb_t b)
#define DECL_mod_1s_2p(name) \
    DECL_mod_1_1p(name)
#define DECL_mod_1s_2p_cps(name) \
    DECL_mod_1_1p_cps(name)
#define DECL_mod_1s_4p(name) \
    DECL_mod_1_1p(name)
#define DECL_mod_1s_4p_cps(name) \
    DECL_mod_1_1p_cps(name)
#define DECL_mod_34lsub1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_srcptr, mp_size_t)
#define DECL_modexact_1c_odd(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t)
#define DECL_mul_1(name) \
    DECL_addmul_1(name)
#define DECL_mul_basecase(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void name(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t)
#define DECL_mullo_basecase(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void name(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_preinv_divrem_1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int)
#define DECL_preinv_mod_1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t)
#define DECL_redc_1(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_redc_2(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t name(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr)
#define DECL_rshift(name) \
    DECL_lshift(name)
#define DECL_sqr_basecase(name) \
    __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void name(mp_ptr, mp_srcptr, mp_size_t)
#define DECL_sub_n(name) \
    DECL_add_n(name)
#define DECL_sublsh1_n(name) \
    DECL_add_n(name)
#define DECL_submul_1(name) \
    DECL_addmul_1(name)

//----------------------------------//
//--------GPGMP MODIFICATIONS-------//
//----------------------------------//

//Helpful macro to determine whether to 'make a fuss' when dynamic memory is allocated - by default we do this inside any CUDA code since dynamic allocation is awful practice in CUDA kernels...
#ifdef __CUDA_ARCH__
#define FUSS_WHEN_DYNAMIC_ALLOCATING 1
#else
#define FUSS_WHEN_DYNAMIC_ALLOCATING 0
#endif

//----------------------------------//
//--------GMP FILE INLININGS--------//
//----------------------------------//

#if !defined(__GMP_WITHIN_CONFIGURE)
#include "config.cuh"
#include "gmp.h"
#include "gpgmp-gmp.cuh"

//=========================GMP-MPARAM.H
//(No content in generic gmp-mparam.h file...)

//=========================FIB_TABLE.H
#if GMP_NUMB_BITS != 64
    Error, error, this data is for 64 bits
#endif

#define FIB_TABLE_LIMIT 93
#define FIB_TABLE_LUCNUM_LIMIT 92

    //==========================FAC_TABLE.H
#if GMP_NUMB_BITS != 64
    Error , error this data is for 64 GMP_NUMB_BITS only
#endif
    /* This table is 0!,1!,2!,3!,...,n! where n! has <= GMP_NUMB_BITS bits */
#define ONE_LIMB_FACTORIAL_TABLE CNST_LIMB(0x1), CNST_LIMB(0x1), CNST_LIMB(0x2), CNST_LIMB(0x6), CNST_LIMB(0x18), CNST_LIMB(0x78), CNST_LIMB(0x2d0), CNST_LIMB(0x13b0), CNST_LIMB(0x9d80), CNST_LIMB(0x58980), CNST_LIMB(0x375f00), CNST_LIMB(0x2611500), CNST_LIMB(0x1c8cfc00), CNST_LIMB(0x17328cc00), CNST_LIMB(0x144c3b2800), CNST_LIMB(0x13077775800), CNST_LIMB(0x130777758000), CNST_LIMB(0x1437eeecd8000), CNST_LIMB(0x16beecca730000), CNST_LIMB(0x1b02b9306890000), CNST_LIMB(0x21c3677c82b40000)

    /* This table is 0!,1!,2!/2,3!/2,...,n!/2^sn where n!/2^sn is an */
    /* odd integer for each n, and n!/2^sn has <= GMP_NUMB_BITS bits */
#define ONE_LIMB_ODD_FACTORIAL_TABLE CNST_LIMB(0x1), CNST_LIMB(0x1), CNST_LIMB(0x1), CNST_LIMB(0x3), CNST_LIMB(0x3), CNST_LIMB(0xf), CNST_LIMB(0x2d), CNST_LIMB(0x13b), CNST_LIMB(0x13b), CNST_LIMB(0xb13), CNST_LIMB(0x375f), CNST_LIMB(0x26115), CNST_LIMB(0x7233f), CNST_LIMB(0x5cca33), CNST_LIMB(0x2898765), CNST_LIMB(0x260eeeeb), CNST_LIMB(0x260eeeeb), CNST_LIMB(0x286fddd9b), CNST_LIMB(0x16beecca73), CNST_LIMB(0x1b02b930689), CNST_LIMB(0x870d9df20ad), CNST_LIMB(0xb141df4dae31), CNST_LIMB(0x79dd498567c1b), CNST_LIMB(0xaf2e19afc5266d), CNST_LIMB(0x20d8a4d0f4f7347), CNST_LIMB(0x335281867ec241ef)
#define ODD_FACTORIAL_TABLE_MAX CNST_LIMB(0x335281867ec241ef)
#define ODD_FACTORIAL_TABLE_LIMIT (25)

    /* Previous table, continued, values modulo 2^GMP_NUMB_BITS */
#define ONE_LIMB_ODD_FACTORIAL_EXTTABLE CNST_LIMB(0x9b3093d46fdd5923), CNST_LIMB(0x5e1f9767cc5866b1), CNST_LIMB(0x92dd23d6966aced7), CNST_LIMB(0xa30d0f4f0a196e5b), CNST_LIMB(0x8dc3e5a1977d7755), CNST_LIMB(0x2ab8ce915831734b), CNST_LIMB(0x2ab8ce915831734b), CNST_LIMB(0x81d2a0bc5e5fdcab), CNST_LIMB(0x9efcac82445da75b), CNST_LIMB(0xbc8b95cf58cde171), CNST_LIMB(0xa0e8444a1f3cecf9), CNST_LIMB(0x4191deb683ce3ffd), CNST_LIMB(0xddd3878bc84ebfc7), CNST_LIMB(0xcb39a64b83ff3751), CNST_LIMB(0xf8203f7993fc1495), CNST_LIMB(0xbd2a2a78b35f4bdd), CNST_LIMB(0x84757be6b6d13921), CNST_LIMB(0x3fbbcfc0b524988b), CNST_LIMB(0xbd11ed47c8928df9), CNST_LIMB(0x3c26b59e41c2f4c5), CNST_LIMB(0x677a5137e883fdb3), CNST_LIMB(0xff74e943b03b93dd), CNST_LIMB(0xfe5ebbcb10b2bb97), CNST_LIMB(0xb021f1de3235e7e7), CNST_LIMB(0x33509eb2e743a58f), CNST_LIMB(0x390f9da41279fb7d), CNST_LIMB(0xe5cb0154f031c559), CNST_LIMB(0x93074695ba4ddb6d), CNST_LIMB(0x81c471caa636247f), CNST_LIMB(0xe1347289b5a1d749), CNST_LIMB(0x286f21c3f76ce2ff), CNST_LIMB(0xbe84a2173e8ac7), CNST_LIMB(0x1595065ca215b88b), CNST_LIMB(0xf95877595b018809), CNST_LIMB(0x9c2efe3c5516f887), CNST_LIMB(0x373294604679382b), CNST_LIMB(0xaf1ff7a888adcd35), CNST_LIMB(0x18ddf279a2c5800b), CNST_LIMB(0x18ddf279a2c5800b), CNST_LIMB(0x505a90e2542582cb), CNST_LIMB(0x5bacad2cd8d5dc2b), CNST_LIMB(0xfe3152bcbff89f41)
#define ODD_FACTORIAL_EXTTABLE_LIMIT (67)

    /* This table is 1!!,3!!,...,(2n+1)!! where (2n+1)!! has <= GMP_NUMB_BITS bits */
#define ONE_LIMB_ODD_DOUBLEFACTORIAL_TABLE CNST_LIMB(0x1), CNST_LIMB(0x3), CNST_LIMB(0xf), CNST_LIMB(0x69), CNST_LIMB(0x3b1), CNST_LIMB(0x289b), CNST_LIMB(0x20fdf), CNST_LIMB(0x1eee11), CNST_LIMB(0x20dcf21), CNST_LIMB(0x27065f73), CNST_LIMB(0x33385d46f), CNST_LIMB(0x49a10615f9), CNST_LIMB(0x730b9982551), CNST_LIMB(0xc223930bef8b), CNST_LIMB(0x15fe07a85a22bf), CNST_LIMB(0x2a9c2ed62ea3521), CNST_LIMB(0x57e22099c030d941)
#define ODD_DOUBLEFACTORIAL_TABLE_MAX CNST_LIMB(0x57e22099c030d941)
#define ODD_DOUBLEFACTORIAL_TABLE_LIMIT (33)

    /* This table x_1, x_2,... contains values s.t. x_n^n has <= GMP_NUMB_BITS bits */
#define NTH_ROOT_NUMB_MASK_TABLE (GMP_NUMB_MASK), CNST_LIMB(0xffffffff), CNST_LIMB(0x285145), CNST_LIMB(0xffff), CNST_LIMB(0x1bdb), CNST_LIMB(0x659), CNST_LIMB(0x235), CNST_LIMB(0xff)

    /* This table contains inverses of odd factorials, modulo 2^GMP_NUMB_BITS */

    /* It begins with (2!/2)^-1=1 */
#define ONE_LIMB_ODD_FACTORIAL_INVERSES_TABLE CNST_LIMB(0x1), CNST_LIMB(0xaaaaaaaaaaaaaaab), CNST_LIMB(0xaaaaaaaaaaaaaaab), CNST_LIMB(0xeeeeeeeeeeeeeeef), CNST_LIMB(0x4fa4fa4fa4fa4fa5), CNST_LIMB(0x2ff2ff2ff2ff2ff3), CNST_LIMB(0x2ff2ff2ff2ff2ff3), CNST_LIMB(0x938cc70553e3771b), CNST_LIMB(0xb71c27cddd93e49f), CNST_LIMB(0xb38e3229fcdee63d), CNST_LIMB(0xe684bb63544a4cbf), CNST_LIMB(0xc2f684917ca340fb), CNST_LIMB(0xf747c9cba417526d), CNST_LIMB(0xbb26eb51d7bd49c3), CNST_LIMB(0xbb26eb51d7bd49c3), CNST_LIMB(0xb0a7efb985294093), CNST_LIMB(0xbe4b8c69f259eabb), CNST_LIMB(0x6854d17ed6dc4fb9), CNST_LIMB(0xe1aa904c915f4325), CNST_LIMB(0x3b8206df131cead1), CNST_LIMB(0x79c6009fea76fe13), CNST_LIMB(0xd8c5d381633cd365), CNST_LIMB(0x4841f12b21144677), CNST_LIMB(0x4a91ff68200b0d0f), CNST_LIMB(0x8f9513a58c4f9e8b), CNST_LIMB(0x2b3e690621a42251), CNST_LIMB(0x4f520f00e03c04e7), CNST_LIMB(0x2edf84ee600211d3), CNST_LIMB(0xadcaa2764aaacdfd), CNST_LIMB(0x161f4f9033f4fe63), CNST_LIMB(0x161f4f9033f4fe63), CNST_LIMB(0xbada2932ea4d3e03), CNST_LIMB(0xcec189f3efaa30d3), CNST_LIMB(0xf7475bb68330bf91), CNST_LIMB(0x37eb7bf7d5b01549), CNST_LIMB(0x46b35660a4e91555), CNST_LIMB(0xa567c12d81f151f7), CNST_LIMB(0x4c724007bb2071b1), CNST_LIMB(0xf4a0cce58a016bd), CNST_LIMB(0xfa21068e66106475), CNST_LIMB(0x244ab72b5a318ae1), CNST_LIMB(0x366ce67e080d0f23), CNST_LIMB(0xd666fdae5dd2a449), CNST_LIMB(0xd740ddd0acc06a0d), CNST_LIMB(0xb050bbbb28e6f97b), CNST_LIMB(0x70b003fe890a5c75), CNST_LIMB(0xd03aabff83037427), CNST_LIMB(0x13ec4ca72c783bd7), CNST_LIMB(0x90282c06afdbd96f), CNST_LIMB(0x4414ddb9db4a95d5), CNST_LIMB(0xa2c68735ae6832e9), CNST_LIMB(0xbf72d71455676665), CNST_LIMB(0xa8469fab6b759b7f), CNST_LIMB(0xc1e55b56e606caf9), CNST_LIMB(0x40455630fc4a1cff), CNST_LIMB(0x120a7b0046d16f7), CNST_LIMB(0xa7c3553b08faef23), CNST_LIMB(0x9f0bfd1b08d48639), CNST_LIMB(0xa433ffce9a304d37), CNST_LIMB(0xa22ad1d53915c683), CNST_LIMB(0xcb6cbc723ba5dd1d), CNST_LIMB(0x547fb1b8ab9d0ba3), CNST_LIMB(0x547fb1b8ab9d0ba3), CNST_LIMB(0x8f15a826498852e3)

    /* This table contains 2n-popc(2n) for small n */

    /* It begins with 2-1=1 (n=1) */
#define TABLE_2N_MINUS_POPC_2N 1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42, 46, 47, 49, 50, 53, 54, 56, 57, 63, 64, 66, 67, 70, 71, 73, 74, 78
#define TABLE_LIMIT_2N_MINUS_POPC_2N 81
#define ODD_CENTRAL_BINOMIAL_OFFSET (13)

    /* This table contains binomial(2k,k)/2^t */

    /* It begins with ODD_CENTRAL_BINOMIAL_TABLE_MIN */
#define ONE_LIMB_ODD_CENTRAL_BINOMIAL_TABLE CNST_LIMB(0x13d66b), CNST_LIMB(0x4c842f), CNST_LIMB(0x93ee7d), CNST_LIMB(0x11e9e123), CNST_LIMB(0x22c60053), CNST_LIMB(0x873ae4d1), CNST_LIMB(0x10757bd97), CNST_LIMB(0x80612c6cd), CNST_LIMB(0xfaa556bc1), CNST_LIMB(0x3d3cc24821), CNST_LIMB(0x77cfeb6bbb), CNST_LIMB(0x7550ebd97c7), CNST_LIMB(0xe5f08695caf), CNST_LIMB(0x386120ffce11), CNST_LIMB(0x6eabb28dd6df), CNST_LIMB(0x3658e31c82a8f), CNST_LIMB(0x6ad2050312783), CNST_LIMB(0x1a42902a5af0bf), CNST_LIMB(0x33ac44f881661d), CNST_LIMB(0xcb764f927d82123), CNST_LIMB(0x190c23fa46b93983), CNST_LIMB(0x62b7609e25caf1b9), CNST_LIMB(0xc29cb72925ef2cff)
#define ODD_CENTRAL_BINOMIAL_TABLE_LIMIT (35)

    /* This table contains the inverses of elements in the previous table. */
#define ONE_LIMB_ODD_CENTRAL_BINOMIAL_INVERSE_TABLE CNST_LIMB(0x61e5bd199bb12643), CNST_LIMB(0x78321494dc8342cf), CNST_LIMB(0x4fd348704ebf7ad5), CNST_LIMB(0x7e722ba086ab568b), CNST_LIMB(0xa5fcc124265843db), CNST_LIMB(0x89c4a6b18633f431), CNST_LIMB(0x4daa2c15f8ce9227), CNST_LIMB(0x801c618ca9be9605), CNST_LIMB(0x32dc192f948a441), CNST_LIMB(0xd02b90c2bf3be1), CNST_LIMB(0xd897e8c1749aa173), CNST_LIMB(0x54a234fc01fef9f7), CNST_LIMB(0x83ff2ab4d1ff7a4f), CNST_LIMB(0xa427f1c9b304e2f1), CNST_LIMB(0x9c14595d1793651f), CNST_LIMB(0x883a71c607a7b46f), CNST_LIMB(0xd089863c54bc9f2b), CNST_LIMB(0x9022f6bce5d07f3f), CNST_LIMB(0xbec207e218768c35), CNST_LIMB(0x9d70cb4cbb4f168b), CNST_LIMB(0x3c3d3403828a9d2b), CNST_LIMB(0x7672df58c56bc489), CNST_LIMB(0x1e66ca55d727d2ff)

    /* This table contains the values t in the formula binomial(2k,k)/2^t */
#define CENTRAL_BINOMIAL_2FAC_TABLE 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3

    //====================SIEVE_TABLE.H
#if GMP_LIMB_BITS != 64
    Error, error, this data is for 64 bits
#endif

#define PRIMESIEVE_INIT_TABLE                                        \
        CNST_LIMB(0x3294C9E069128480),     /* 5 - 196 (42 primes) */     \
            CNST_LIMB(0x95A35E1EC4AB21DC), /* 197 - 388 (32 primes) */   \
            CNST_LIMB(0x4AD7CE99B8693366), /* 389 - 580 (30 primes) */   \
            CNST_LIMB(0x6595B6DA728DC52B), /* 581 - 772 (30 primes) */   \
            CNST_LIMB(0xEA6D9F8787B0CEDE), /* 773 - 964 (26 primes) */   \
            CNST_LIMB(0x3F56A1F4CD3275A9), /* 965 - 1156 (29 primes) */  \
            CNST_LIMB(0xFD3848FB74A76ADB), /* 1157 - 1348 (26 primes) */ \
            CNST_LIMB(0xDBBA0DD1A1EDF6AF), /* 1349 - 1540 (25 primes) */ \
            CNST_LIMB(0xCEC7F17ED22799A5), /* 1541 - 1732 (27 primes) */ \
            CNST_LIMB(0xEAEC17BDBB717D56), /* 1733 - 1924 (24 primes) */ \
            CNST_LIMB(0x3B0EB7B3585AFCF3), /* 1925 - 2116 (26 primes) */ \
            CNST_LIMB(0xE563D8F69FDF6C4F), /* 2117 - 2308 (23 primes) */ \
            CNST_LIMB(0xFE5BA7ABA45E92FC), /* 2309 - 2500 (25 primes) */ \
            CNST_LIMB(0x158DEE6F3BF49B7D), /* 2501 - 2692 (24 primes) */ \
            CNST_LIMB(0xBE5A7BC4EDE6CD1A), /* 2693 - 2884 (26 primes) */ \
            CNST_LIMB(0xD7679B3FCA7BB6AD), /* 2885 - 3076 (22 primes) */ \
            CNST_LIMB(0xC3F66B971FEF37E9), /* 3077 - 3268 (22 primes) */ \
            CNST_LIMB(0x6F7EBCF339C953FD), /* 3269 - 3460 (22 primes) */ \
            CNST_LIMB(0xD5A5ECDCD235DBF0), /* 3461 - 3652 (27 primes) */ \
            CNST_LIMB(0xECFA7B2FD5B65E3B), /* 3653 - 3844 (22 primes) */ \
            CNST_LIMB(0xD28EFDF9C89F67B1), /* 3845 - 4036 (25 primes) */ \
            CNST_LIMB(0xCB7F7C7A3DD3AF4F), /* 4037 - 4228 (21 primes) */ \
            CNST_LIMB(0xEEBED6CDFF6B32CC), /* 4229 - 4420 (22 primes) */ \
            CNST_LIMB(0xD5BD73F85ECFA97C), /* 4421 - 4612 (23 primes) */ \
            CNST_LIMB(0x21FDBE4FBBAD48F7), /* 4613 - 4804 (24 primes) */ \
            CNST_LIMB(0x5E35A3B5EEB7FDE7), /* 4805 - 4996 (21 primes) */ \
            CNST_LIMB(0xD9EBFD53A7DBBCC9), /* 4997 - 5188 (22 primes) */ \
            CNST_LIMB(0xFF9EDEAF2EFE1F76), /* 5189 - 5380 (18 primes) */
#define PRIMESIEVE_NUMBEROF_TABLE 28
    /* #define PRIMESIEVE_PRIMES_IN_TABLE 706 */
#define PRIMESIEVE_HIGHEST_PRIME 5351
    /* #define PRIMESIEVE_FIRST_UNCHECKED 5381 */

#define SIEVE_MASK1 CNST_LIMB(0x3204C1A049120485)
#define SIEVE_MASKT CNST_LIMB(0xA1204892058)
#define SIEVE_2MSK1 CNST_LIMB(0x29048402110840A)
#define SIEVE_2MSK2 CNST_LIMB(0x9402180C40230184)
#define SIEVE_2MSKT CNST_LIMB(0x5021088402120)

    //==============================MP_BASES.H
#if GMP_NUMB_BITS != 64
    Error, error, this data is for 64 bits
#endif

    /* mp_bases[10] data, as literal values */
#define MP_BASES_CHARS_PER_LIMB_10 19
#define MP_BASES_BIG_BASE_CTZ_10 19
#define MP_BASES_BIG_BASE_10 CNST_LIMB(0x8ac7230489e80000)
#define MP_BASES_BIG_BASE_INVERTED_10 CNST_LIMB(0xd83c94fb6d2ac34a)
#define MP_BASES_BIG_BASE_BINVERTED_10 CNST_LIMB(0x26b172506559ce15)
#define MP_BASES_NORMALIZATION_STEPS_10 0

#if WANT_FAT_BINARY
#include "fat.h"
#endif
#endif

#if HAVE_INTTYPES_H /* for uint_least32_t */
#include <inttypes.h>
#endif
    /* On some platforms inttypes.h exists but is incomplete
    and we still need stdint.h. */
#if HAVE_STDINT_H
#include <stdint.h>
#endif

#ifdef __cplusplus
#include <cstring> /* for strlen */
#include <string>  /* for std::string */
#endif

    //----------------------------------//
    //------EOF GMP FILE INLININGS------//
    //----------------------------------//

#include "Definitions.cuh"

//Let's override some basic GMP defines so this file can mesh with GPGMP instead of GMP.

namespace gpgmp
    {
        namespace mpnRoutines
        {

        }
    }



#define GPGMP_MPN_NAMESPACE_BEGIN \
        namespace gpgmp               \
        {                             \
            namespace mpnRoutines     \
            {
#define GPGMP_MPN_NAMESPACE_END \
        }                           \
        }

    //----------------------------------//
    //------EOF GPGMP MODIFICATIONS-----//
    //----------------------------------//

#ifndef WANT_TMP_DEBUG /* for TMP_ALLOC_LIMBS_2 and others */
#define WANT_TMP_DEBUG 0
#endif

    /* The following tries to get a good version of alloca.  The tests are
       adapted from autoconf AC_FUNC_ALLOCA, with a couple of additions.
       Whether this succeeds is tested by GMP_FUNC_ALLOCA and HAVE_ALLOCA will
       be setup appropriately.

       ifndef alloca - a cpp define might already exist.
           glibc <stdlib.h> includes <alloca.h> which uses GCC __builtin_alloca.
           HP cc +Olibcalls adds a #define of alloca to __builtin_alloca.

       GCC __builtin_alloca - preferred whenever available.

       _AIX pragma - IBM compilers need a #pragma in "each module that needs to
           use alloca".  Pragma indented to protect pre-ANSI cpp's.  _IBMR2 was
           used in past versions of GMP, retained still in case it matters.

           The autoconf manual says this pragma needs to be at the start of a C
           file, apart from comments and preprocessor directives.  Is that true?
           xlc on aix 4.xxx doesn't seem to mind it being after prototypes etc
           from gmp.h.
    */

    // Yes, I just wrapped the entirety of gmp-impl.h in a namespace, among other things, to get gpgmp's base working.
    // Yes, I vomitted in my mouth doing so.
    // No, I am not going to approach this differently unless I need to...

#ifndef alloca
#ifdef __GNUC__
#define alloca __builtin_alloca
#else
#ifdef __DECC
#define alloca(x) __ALLOCA(x)
#else
#ifdef _MSC_VER
#include <malloc.h>
#define alloca _alloca
#else
#if HAVE_ALLOCA_H
#include <alloca.h>
#else
#if defined(_AIX) || defined(_IBMR2)
#pragma alloca
#else
char *alloca();
#endif
#endif
#endif
#endif
#endif
#endif

    /* if not provided by gmp-mparam.h */
#ifndef GMP_LIMB_BYTES
#define GMP_LIMB_BYTES SIZEOF_MP_LIMB_T
#endif
#ifndef GMP_LIMB_BITS
#define GMP_LIMB_BITS (8 * SIZEOF_MP_LIMB_T)
#endif

#define BITS_PER_ULONG (8 * SIZEOF_UNSIGNED_LONG)

    /* gmp_uint_least32_t is an unsigned integer type with at least 32 bits. */
#if HAVE_UINT_LEAST32_T
    typedef uint_least32_t gmp_uint_least32_t;
#else
#if SIZEOF_UNSIGNED_SHORT >= 4
typedef unsigned short gmp_uint_least32_t;
#else
#if SIZEOF_UNSIGNED >= 4
typedef unsigned gmp_uint_least32_t;
#else
typedef unsigned long gmp_uint_least32_t;
#endif
#endif
#endif

    /* gmp_intptr_t, for pointer to integer casts */
#if HAVE_INTPTR_T
    typedef intptr_t gmp_intptr_t;
#else /* fallback */
typedef size_t gmp_intptr_t;
#endif

    /* pre-inverse types for truncating division and modulo */
    typedef struct
    {
        mp_limb_t inv32;
    } gmp_pi1_t;
    typedef struct
    {
        mp_limb_t inv21, inv32, inv53;
    } gmp_pi2_t;

    /* "const" basically means a function does nothing but examine its arguments
       and give a return value, it doesn't read or write any memory (neither
       global nor pointed to by arguments), and has no other side-effects.  This
       is more restrictive than "pure".  See info node "(gcc)Function
       Attributes".  __GMP_NO_ATTRIBUTE_CONST_PURE lets tune/common.c etc turn
       this off when trying to write timing loops.  */
#if HAVE_ATTRIBUTE_CONST && !defined(__GMP_NO_ATTRIBUTE_CONST_PURE)
#define ATTRIBUTE_CONST __attribute__((const))
#else
#define ATTRIBUTE_CONST
#endif

#if HAVE_ATTRIBUTE_NORETURN
#define ATTRIBUTE_NORETURN __attribute__((noreturn))
#else
#define ATTRIBUTE_NORETURN
#endif

    /* "malloc" means a function behaves like malloc in that the pointer it
       returns doesn't alias anything.  */
#if HAVE_ATTRIBUTE_MALLOC
#define ATTRIBUTE_MALLOC __attribute__((malloc))
#else
#define ATTRIBUTE_MALLOC
#endif

#if !HAVE_STRCHR
#define strchr(s, c) index(s, c)
#endif

#if !HAVE_MEMSET
#define memset(p, c, n)                 \
        do                                  \
        {                                   \
            ASSERT((n) >= 0);               \
            char *__memset__p = (p);        \
            int __i;                        \
            for (__i = 0; __i < (n); __i++) \
                __memset__p[__i] = (c);     \
        } while (0)
#endif

    /* va_copy is standard in C99, and gcc provides __va_copy when in strict C89
       mode.  Falling back to a memcpy will give maximum portability, since it
       works no matter whether va_list is a pointer, struct or array.  */
#if !defined(va_copy) && defined(__va_copy)
#define va_copy(dst, src) __va_copy(dst, src)
#endif
#if !defined(va_copy)
#define va_copy(dst, src)                        \
        do                                           \
        {                                            \
            memcpy(&(dst), &(src), sizeof(va_list)); \
        } while (0)
#endif

    /* HAVE_HOST_CPU_alpha_CIX is 1 on an alpha with the CIX instructions
       (ie. ctlz, ctpop, cttz).  */
#if HAVE_HOST_CPU_alphaev67 || HAVE_HOST_CPU_alphaev68 || HAVE_HOST_CPU_alphaev7
#define HAVE_HOST_CPU_alpha_CIX 1
#endif

#if defined(__cplusplus)
    extern "C"
    {
#endif

        /* Usage: TMP_DECL;
              TMP_MARK;
              ptr = TMP_ALLOC (bytes);
              TMP_FREE;

           Small allocations should use TMP_SALLOC, big allocations should use
           TMP_BALLOC.  Allocations that might be small or big should use TMP_ALLOC.

           Functions that use just TMP_SALLOC should use TMP_SDECL, TMP_SMARK, and
           TMP_SFREE.

           TMP_DECL just declares a variable, but might be empty and so must be last
           in a list of variables.  TMP_MARK must be done before any TMP_ALLOC.
           TMP_ALLOC(0) is not allowed.  TMP_FREE doesn't need to be done if a
           TMP_MARK was made, but then no TMP_ALLOCs.  */

        /* The alignment in bytes, used for TMP_ALLOCed blocks, when alloca or
           __gpgmp_allocate_func doesn't already determine it.  */
        union tmp_align_t
        {
            mp_limb_t l;
            double d;
            char *p;
        };
#define __TMP_ALIGN sizeof(union tmp_align_t)

        /* Return "a" rounded upwards to a multiple of "m", if it isn't already.
           "a" must be an unsigned type.
           This is designed for use with a compile-time constant "m".
           The POW2 case is expected to be usual, and gcc 3.0 and up recognises
           "(-(8*n))%8" or the like is always zero, which means the rounding up in
           the WANT_TMP_NOTREENTRANT version of TMP_ALLOC below will be a noop.  */
#define ROUND_UP_MULTIPLE(a, m)     \
        (POW2_P(m) ? (a) + (-(a)) % (m) \
                   : (a) + (m) - 1 - (((a) + (m) - 1) % (m)))

#if defined(WANT_TMP_ALLOCA) || defined(WANT_TMP_REENTRANT)
        struct tmp_reentrant_t
        {
            struct tmp_reentrant_t *next;
            size_t size; /* bytes, including header */
        };

#define HSIZ   ROUND_UP_MULTIPLE (sizeof (struct tmp_reentrant_t), __TMP_ALIGN)



#endif

#if WANT_TMP_ALLOCA
#define TMP_SDECL
#define TMP_DECL struct tmp_reentrant_t *__tmp_marker
#define TMP_SMARK
#define TMP_MARK __tmp_marker = 0
#define TMP_SALLOC(n) alloca(n)
#define TMP_BALLOC(n) __gpgmp_tmp_reentrant_alloc(&__tmp_marker, n)
        /* The peculiar stack allocation limit here is chosen for efficient asm.  */
#define TMP_ALLOC(n) \
        (LIKELY((n) <= 0x7f00) ? TMP_SALLOC(n) : TMP_BALLOC(n))
#define TMP_SFREE
#define TMP_FREE                                    \
        do                                              \
        {                                               \
            if (UNLIKELY(__tmp_marker != 0))            \
                __gpgmp_tmp_reentrant_free(__tmp_marker); \
        } while (0)
#endif

#if WANT_TMP_REENTRANT
#define TMP_SDECL TMP_DECL
#define TMP_DECL struct tmp_reentrant_t *__tmp_marker
#define TMP_SMARK TMP_MARK
#define TMP_MARK __tmp_marker = 0
#define TMP_SALLOC(n) TMP_ALLOC(n)
#define TMP_BALLOC(n) TMP_ALLOC(n)
#define TMP_ALLOC(n) __gpgmp_tmp_reentrant_alloc(&__tmp_marker, n)
#define TMP_SFREE TMP_FREE
#define TMP_FREE __gpgmp_tmp_reentrant_free(__tmp_marker)
#endif

#if WANT_TMP_NOTREENTRANT
        struct tmp_marker
        {
            struct tmp_stack *which_chunk;
            void *alloc_point;
        };
        GPGMP_MPN_NAMESPACE_BEGIN
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void *__gmp_tmp_alloc(unsigned long) ATTRIBUTE_MALLOC;
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gmp_tmp_mark(struct tmp_marker *);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gmp_tmp_free(struct tmp_marker *);
        GPGMP_MPN_NAMESPACE_END
#define TMP_SDECL TMP_DECL
#define TMP_DECL struct tmp_marker __tmp_marker
#define TMP_SMARK TMP_MARK
#define TMP_MARK __gmp_tmp_mark(&__tmp_marker)
#define TMP_SALLOC(n) TMP_ALLOC(n)
#define TMP_BALLOC(n) TMP_ALLOC(n)
#define TMP_ALLOC(n) \
        __gmp_tmp_alloc(ROUND_UP_MULTIPLE((unsigned long)(n), __TMP_ALIGN))
#define TMP_SFREE TMP_FREE
#define TMP_FREE __gmp_tmp_free(&__tmp_marker)
#endif

#if WANT_TMP_DEBUG
        /* See tal-debug.c for some comments. */
        struct tmp_debug_t
        {
            struct tmp_debug_entry_t *list;
            const char *file;
            int line;
        };
        struct tmp_debug_entry_t
        {
            struct tmp_debug_entry_t *next;
            void *block;
            size_t size;
        };
        GPGMP_MPN_NAMESPACE_BEGIN
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gmp_tmp_debug_mark(const char *, int, struct tmp_debug_t **,
                                                                      struct tmp_debug_t *,
                                                                      const char *, const char *);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void *__gmp_tmp_debug_alloc(const char *, int, int,
                                                                        struct tmp_debug_t **, const char *,
                                                                        size_t) ATTRIBUTE_MALLOC;
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gmp_tmp_debug_free(const char *, int, int,
                                                                      struct tmp_debug_t **,
                                                                      const char *, const char *);
        GPGMP_MPN_NAMESPACE_END
#define TMP_SDECL TMP_DECL_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_DECL TMP_DECL_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_SMARK TMP_MARK_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_MARK TMP_MARK_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_SFREE TMP_FREE_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_FREE TMP_FREE_NAME(__tmp_xmarker, "__tmp_marker")
        /* The marker variable is designed to provoke an uninitialized variable
           warning from the compiler if TMP_FREE is used without a TMP_MARK.
           __tmp_marker_inscope does the same for TMP_ALLOC.  Runtime tests pick
           these things up too.  */
#define TMP_DECL_NAME(marker, marker_name)       \
        int marker;                                  \
        int __tmp_marker_inscope;                    \
        const char *__tmp_marker_name = marker_name; \
        struct tmp_debug_t __tmp_marker_struct;      \
        /* don't demand NULL, just cast a zero */    \
        struct tmp_debug_t *__tmp_marker = (struct tmp_debug_t *)0
#define TMP_MARK_NAME(marker, marker_name)                        \
        do                                                            \
        {                                                             \
            marker = 1;                                               \
            __tmp_marker_inscope = 1;                                 \
            __gmp_tmp_debug_mark(ASSERT_FILE, ASSERT_LINE,            \
                                 &__tmp_marker, &__tmp_marker_struct, \
                                 __tmp_marker_name, marker_name);     \
        } while (0)
#define TMP_SALLOC(n) TMP_ALLOC(n)
#define TMP_BALLOC(n) TMP_ALLOC(n)
#define TMP_ALLOC(size)                             \
        __gmp_tmp_debug_alloc(ASSERT_FILE, ASSERT_LINE, \
                              __tmp_marker_inscope,     \
                              &__tmp_marker, __tmp_marker_name, size)
#define TMP_FREE_NAME(marker, marker_name)                    \
        do                                                        \
        {                                                         \
            __gmp_tmp_debug_free(ASSERT_FILE, ASSERT_LINE,        \
                                 marker, &__tmp_marker,           \
                                 __tmp_marker_name, marker_name); \
        } while (0)
#endif /* WANT_TMP_DEBUG */

        /* Allocating various types. */


#if FUSS_WHEN_DYNAMIC_ALLOCATING
#define TMP_ALLOC_TYPE(n, type) (printf("TMP_ALLOC_TYPE Dynamic Allocation is being performed, this should not happen in a GPU environment, refactoring needed...\n") ? ((type *)TMP_ALLOC((n) * sizeof(type))) : ((type *)TMP_ALLOC((n) * sizeof(type))))
#define TMP_SALLOC_TYPE(n, type) (printf("TMP_SALLOC_TYPE Dynamic Allocation is being performed, this should not happen in a GPU environment, refactoring needed...\n") ? ((type *)TMP_SALLOC((n) * sizeof(type))) : ((type *)TMP_SALLOC((n) * sizeof(type))))
#define TMP_BALLOC_TYPE(n, type) (printf("TMP_BALLOC_TYPE Dynamic Allocation is being performed, this should not happen in a GPU environment, refactoring needed...\n") ? ((type *)TMP_BALLOC((n) * sizeof(type))) : ((type *)TMP_BALLOC((n) * sizeof(type))))
#else
#define TMP_ALLOC_TYPE(n, type) ((type *)TMP_ALLOC((n) * sizeof(type)))
#define TMP_SALLOC_TYPE(n, type) ((type *)TMP_SALLOC((n) * sizeof(type)))
#define TMP_BALLOC_TYPE(n, type) ((type *)TMP_BALLOC((n) * sizeof(type)))
#endif
#define TMP_ALLOC_LIMBS(n) TMP_ALLOC_TYPE(n, mp_limb_t)
#define TMP_SALLOC_LIMBS(n) TMP_SALLOC_TYPE(n, mp_limb_t)
#define TMP_BALLOC_LIMBS(n) TMP_BALLOC_TYPE(n, mp_limb_t)
#define TMP_ALLOC_MP_PTRS(n) TMP_ALLOC_TYPE(n, mp_ptr)
#define TMP_SALLOC_MP_PTRS(n) TMP_SALLOC_TYPE(n, mp_ptr)
#define TMP_BALLOC_MP_PTRS(n) TMP_BALLOC_TYPE(n, mp_ptr)

        /* It's more efficient to allocate one block than many.  This is certainly
           true of the malloc methods, but it can even be true of alloca if that
           involves copying a chunk of stack (various RISCs), or a call to a stack
           bounds check (mingw).  In any case, when debugging keep separate blocks
           so a redzoning malloc debugger can protect each individually.  */
#define TMP_ALLOC_LIMBS_2(xp, xsize, yp, ysize)        \
        do                                                 \
        {                                                  \
            if (WANT_TMP_DEBUG)                            \
            {                                              \
                (xp) = TMP_ALLOC_LIMBS(xsize);             \
                (yp) = TMP_ALLOC_LIMBS(ysize);             \
            }                                              \
            else                                           \
            {                                              \
                (xp) = TMP_ALLOC_LIMBS((xsize) + (ysize)); \
                (yp) = (xp) + (xsize);                     \
            }                                              \
        } while (0)
#define TMP_ALLOC_LIMBS_3(xp, xsize, yp, ysize, zp, zsize)       \
        do                                                           \
        {                                                            \
            if (WANT_TMP_DEBUG)                                      \
            {                                                        \
                (xp) = TMP_ALLOC_LIMBS(xsize);                       \
                (yp) = TMP_ALLOC_LIMBS(ysize);                       \
                (zp) = TMP_ALLOC_LIMBS(zsize);                       \
            }                                                        \
            else                                                     \
            {                                                        \
                (xp) = TMP_ALLOC_LIMBS((xsize) + (ysize) + (zsize)); \
                (yp) = (xp) + (xsize);                               \
                (zp) = (yp) + (ysize);                               \
            }                                                        \
        } while (0)

        /* From gmp.h, nicer names for internal use. */
#define CRAY_Pragma(str) __GMP_CRAY_Pragma(str)
#define MPN_CMP(result, xp, yp, size) __GMPN_CMP(result, xp, yp, size)
#define LIKELY(cond) __GMP_LIKELY(cond)
#define UNLIKELY(cond) __GMP_UNLIKELY(cond)

#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define SGN(n)  ((n) > 0 ? 1 : (n) < 0 ? -1 : 0)
#define NEG_CAST(T, x) (-(__GMP_CAST(T, (x) + 1) - 1))
#define ABS_CAST(T, x) ((x) >= 0 ? __GMP_CAST(T, x) : NEG_CAST(T, x))
#undef MIN
#define MIN(l, o) ((l) < (o) ? (l) : (o))
#undef MAX
#define MAX(h, i) ((h) > (i) ? (h) : (i))
#define numberof(x) (sizeof(x) / sizeof((x)[0]))

        /* Field access macros.  */
#define SIZ(x) ((x)->_mp_size)
#define ABSIZ(x) ABS(SIZ(x))
#define PTR(x) ((x)->_mp_d)
#define EXP(x) ((x)->_mp_exp)
#define PREC(x) ((x)->_mp_prec)
#define ALLOC(x) ((x)->_mp_alloc)
#define NUM(x) mpq_numref(x)
#define DEN(x) mpq_denref(x)

        /* n-1 inverts any low zeros and the lowest one bit.  If n&(n-1) leaves zero
           then that lowest one bit must have been the only bit set.  n==0 will
           return true though, so avoid that.  */
#define POW2_P(n) (((n) & ((n) - 1)) == 0)

        /* This is intended for constant THRESHOLDs only, where the compiler
           can completely fold the result.  */
#define LOG2C(n)                                                         \
        (((n) >= 0x1) + ((n) >= 0x2) + ((n) >= 0x4) + ((n) >= 0x8) +         \
         ((n) >= 0x10) + ((n) >= 0x20) + ((n) >= 0x40) + ((n) >= 0x80) +     \
         ((n) >= 0x100) + ((n) >= 0x200) + ((n) >= 0x400) + ((n) >= 0x800) + \
         ((n) >= 0x1000) + ((n) >= 0x2000) + ((n) >= 0x4000) + ((n) >= 0x8000))

#define MP_LIMB_T_MAX (~(mp_limb_t)0)

        /* Must cast ULONG_MAX etc to unsigned long etc, since they might not be
           unsigned on a K&R compiler.  In particular the HP-UX 10 bundled K&R cc
           treats the plain decimal values in <limits.h> as signed.  */
#define ULONG_HIGHBIT (ULONG_MAX ^ ((unsigned long)ULONG_MAX >> 1))
#define UINT_HIGHBIT (UINT_MAX ^ ((unsigned)UINT_MAX >> 1))
#define USHRT_HIGHBIT (USHRT_MAX ^ ((unsigned short)USHRT_MAX >> 1))
#define GMP_LIMB_HIGHBIT (MP_LIMB_T_MAX ^ (MP_LIMB_T_MAX >> 1))

#if __GMP_MP_SIZE_T_INT
#define MP_SIZE_T_MAX INT_MAX
#define MP_SIZE_T_MIN INT_MIN
#else
#define MP_SIZE_T_MAX LONG_MAX
#define MP_SIZE_T_MIN LONG_MIN
#endif

        /* mp_exp_t is the same as mp_size_t */
#define MP_EXP_T_MAX MP_SIZE_T_MAX
#define MP_EXP_T_MIN MP_SIZE_T_MIN

#define LONG_HIGHBIT LONG_MIN
#define INT_HIGHBIT INT_MIN
#define SHRT_HIGHBIT SHRT_MIN

#define GMP_NUMB_HIGHBIT (CNST_LIMB(1) << (GMP_NUMB_BITS - 1))

#if GMP_NAIL_BITS == 0
#define GMP_NAIL_LOWBIT CNST_LIMB(0)
#else
#define GMP_NAIL_LOWBIT (CNST_LIMB(1) << GMP_NUMB_BITS)
#endif

#if GMP_NAIL_BITS != 0
        /* Set various *_THRESHOLD values to be used for nails.  Thus we avoid using
           code that has not yet been qualified.  */

#undef DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD 50

#undef DIVREM_1_NORM_THRESHOLD
#undef DIVREM_1_UNNORM_THRESHOLD
#undef MOD_1_NORM_THRESHOLD
#undef MOD_1_UNNORM_THRESHOLD
#undef USE_PREINV_DIVREM_1
#undef DIVREM_2_THRESHOLD
#undef DIVEXACT_1_THRESHOLD
#define DIVREM_1_NORM_THRESHOLD MP_SIZE_T_MAX   /* no preinv */
#define DIVREM_1_UNNORM_THRESHOLD MP_SIZE_T_MAX /* no preinv */
#define MOD_1_NORM_THRESHOLD MP_SIZE_T_MAX      /* no preinv */
#define MOD_1_UNNORM_THRESHOLD MP_SIZE_T_MAX    /* no preinv */
#define USE_PREINV_DIVREM_1 0                   /* no preinv */
#define DIVREM_2_THRESHOLD MP_SIZE_T_MAX        /* no preinv */

        /* mpn/generic/mul_fft.c is not nails-capable. */
#undef MUL_FFT_THRESHOLD
#undef SQR_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD MP_SIZE_T_MAX
#define SQR_FFT_THRESHOLD MP_SIZE_T_MAX
#endif

        /* Swap macros. */

#define MP_LIMB_T_SWAP(x, y)                   \
        do                                         \
        {                                          \
            mp_limb_t __mp_limb_t_swap__tmp = (x); \
            (x) = (y);                             \
            (y) = __mp_limb_t_swap__tmp;           \
        } while (0)
#define MP_SIZE_T_SWAP(x, y)                   \
        do                                         \
        {                                          \
            mp_size_t __mp_size_t_swap__tmp = (x); \
            (x) = (y);                             \
            (y) = __mp_size_t_swap__tmp;           \
        } while (0)

#define MP_PTR_SWAP(x, y)                \
        do                                   \
        {                                    \
            mp_ptr __mp_ptr_swap__tmp = (x); \
            (x) = (y);                       \
            (y) = __mp_ptr_swap__tmp;        \
        } while (0)
#define MP_SRCPTR_SWAP(x, y)                   \
        do                                         \
        {                                          \
            mp_srcptr __mp_srcptr_swap__tmp = (x); \
            (x) = (y);                             \
            (y) = __mp_srcptr_swap__tmp;           \
        } while (0)

#define MPN_PTR_SWAP(xp, xs, yp, ys) \
        do                               \
        {                                \
            MP_PTR_SWAP(xp, yp);         \
            MP_SIZE_T_SWAP(xs, ys);      \
        } while (0)
#define MPN_SRCPTR_SWAP(xp, xs, yp, ys) \
        do                                  \
        {                                   \
            MP_SRCPTR_SWAP(xp, yp);         \
            MP_SIZE_T_SWAP(xs, ys);         \
        } while (0)

#define MPZ_PTR_SWAP(x, y)                 \
        do                                     \
        {                                      \
            mpz_ptr __mpz_ptr_swap__tmp = (x); \
            (x) = (y);                         \
            (y) = __mpz_ptr_swap__tmp;         \
        } while (0)
#define MPZ_SRCPTR_SWAP(x, y)                    \
        do                                           \
        {                                            \
            mpz_srcptr __mpz_srcptr_swap__tmp = (x); \
            (x) = (y);                               \
            (y) = __mpz_srcptr_swap__tmp;            \
        } while (0)

#define MPQ_PTR_SWAP(x, y)                 \
        do                                     \
        {                                      \
            mpq_ptr __mpq_ptr_swap__tmp = (x); \
            (x) = (y);                         \
            (y) = __mpq_ptr_swap__tmp;         \
        } while (0)
#define MPQ_SRCPTR_SWAP(x, y)                    \
        do                                           \
        {                                            \
            mpq_srcptr __mpq_srcptr_swap__tmp = (x); \
            (x) = (y);                               \
            (y) = __mpq_srcptr_swap__tmp;            \
        } while (0)


        // TODO: These need to go away eventually
        /* Enhancement: __gpgmp_allocate_func could have "__attribute__ ((malloc))",
           but current gcc (3.0) doesn't seem to support that.  */
        __GPGMP_DECLSPEC void *__gpgmp_allocate_func(size_t);
        __GPGMP_DECLSPEC void *__gpgmp_reallocate_func(void *, size_t, size_t);
        __GPGMP_DECLSPEC void *__gpgmp_free_func(void *, size_t);


#define __GMP_ALLOCATE_FUNC_TYPE(n, type) \
        ((type *)(*__gpgmp_allocate_func)((n) * sizeof(type)))
#define __GMP_ALLOCATE_FUNC_LIMBS(n) __GMP_ALLOCATE_FUNC_TYPE(n, mp_limb_t)


__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void *__gpgmp_tmp_reentrant_alloc(struct tmp_reentrant_t **, size_t);


__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_tmp_reentrant_free(struct tmp_reentrant_t *);


#define __GMP_REALLOCATE_FUNC_TYPE(p, old_size, new_size, type) \
        ((type *)(*__gpgmp_reallocate_func)(p, (old_size) * sizeof(type), (new_size) * sizeof(type)))
#define __GMP_REALLOCATE_FUNC_LIMBS(p, old_size, new_size) \
        __GMP_REALLOCATE_FUNC_TYPE(p, old_size, new_size, mp_limb_t)

#define __GMP_FREE_FUNC_TYPE(p, n, type) (*__gpgmp_free_func)(p, (n) * sizeof(type))
#define __GMP_FREE_FUNC_LIMBS(p, n) __GMP_FREE_FUNC_TYPE(p, n, mp_limb_t)

#define __GMP_REALLOCATE_FUNC_MAYBE(ptr, oldsize, newsize)           \
        do                                                               \
        {                                                                \
            if ((oldsize) != (newsize))                                  \
                (ptr) = (*__gpgmp_reallocate_func)(ptr, oldsize, newsize); \
        } while (0)

#define __GMP_REALLOCATE_FUNC_MAYBE_TYPE(ptr, oldsize, newsize, type)                                          \
        do                                                                                                         \
        {                                                                                                          \
            if ((oldsize) != (newsize))                                                                            \
                (ptr) = (type *)(*__gpgmp_reallocate_func)(ptr, (oldsize) * sizeof(type), (newsize) * sizeof(type)); \
        } while (0)

        /* Dummy for non-gcc, code involving it will go dead. */
#if !defined(__GNUC__) || __GNUC__ < 2
#define __builtin_constant_p(x) 0
#endif

        /* In gcc 2.96 and up on i386, tail calls are optimized to jumps if the
           stack usage is compatible.  __attribute__ ((regparm (N))) helps by
           putting leading parameters in registers, avoiding extra stack.

           regparm cannot be used with calls going through the PLT, because the
           binding code there may clobber the registers (%eax, %edx, %ecx) used for
           the regparm parameters.  Calls to local (ie. static) functions could
           still use this, if we cared to differentiate locals and globals.

           On athlon-unknown-freebsd4.9 with gcc 3.3.3, regparm cannot be used with
           -p or -pg profiling, since that version of gcc doesn't realize the
           .mcount calls will clobber the parameter registers.  Other systems are
           ok, like debian with glibc 2.3.2 (mcount doesn't clobber), but we don't
           bother to try to detect this.  regparm is only an optimization so we just
           disable it when profiling (profiling being a slowdown anyway).  */

#if HAVE_HOST_CPU_FAMILY_x86 && __GMP_GNUC_PREREQ(2, 96) && !defined(PIC) && !WANT_PROFILING_PROF && !WANT_PROFILING_GPROF
#define USE_LEADING_REGPARM 1
#else
#define USE_LEADING_REGPARM 0
#endif

        /* Macros for altering parameter order according to regparm usage. */
#if USE_LEADING_REGPARM
#define REGPARM_2_1(a, b, x) x, a, b
#define REGPARM_3_1(a, b, c, x) x, a, b, c
#define REGPARM_ATTR(n) __attribute__((regparm(n)))
#else
#define REGPARM_2_1(a, b, x) a, b, x
#define REGPARM_3_1(a, b, c, x) a, b, c, x
#define REGPARM_ATTR(n)
#endif

        /* ASM_L gives a local label for a gcc asm block, for use when temporary
           local labels like "1:" might not be available, which is the case for
           instance on the x86s (the SCO assembler doesn't support them).

           The label generated is made unique by including "%=" which is a unique
           number for each insn.  This ensures the same name can be used in multiple
           asm blocks, perhaps via a macro.  Since jumps between asm blocks are not
           allowed there's no need for a label to be usable outside a single
           block.  */

#define ASM_L(name) LSYM_PREFIX "asm_%=_" #name

#if defined(__GNUC__) && HAVE_HOST_CPU_FAMILY_x86
#if 0
/* FIXME: Check that these actually improve things.
   FIXME: Need a cld after each std.
   FIXME: Can't have inputs in clobbered registers, must describe them as
   dummy outputs, and add volatile. */
#define MPN_COPY_INCR(DST, SRC, N) \
        __asm__("cld\n\trep\n\tmovsl" : : "D"(DST), "S"(SRC), "c"(N) : "cx", "di", "si", "memory")
#define MPN_COPY_DECR(DST, SRC, N) \
        __asm__("std\n\trep\n\tmovsl" : : "D"((DST) + (N) - 1), "S"((SRC) + (N) - 1), "c"(N) : "cx", "di", "si", "memory")
#endif
#endif
        GPGMP_MPN_NAMESPACE_BEGIN
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gmpz_aorsmul_1(REGPARM_3_1(mpz_ptr, mpz_srcptr, mp_limb_t, mp_size_t)) REGPARM_ATTR(1);
#define mpz_aorsmul_1(w, u, v, sub) __gmpz_aorsmul_1(REGPARM_3_1(w, u, v, sub))

#define mpz_n_pow_ui __gmpz_n_pow_ui
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void mpz_n_pow_ui(mpz_ptr, mp_srcptr, mp_size_t, unsigned long);

#define gpmpn_addmul_1c __GPGMP_MPN(addmul_1c)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_1c(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#ifndef gpmpn_addmul_2 /* if not done with cpuvec in a fat binary */
#define gpmpn_addmul_2 __GPGMP_MPN(addmul_2)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_2(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
#endif

#define gpmpn_addmul_3 __GPGMP_MPN(addmul_3)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_3(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_addmul_4 __GPGMP_MPN(addmul_4)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_4(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_addmul_5 __GPGMP_MPN(addmul_5)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_5(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_addmul_6 __GPGMP_MPN(addmul_6)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_6(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_addmul_7 __GPGMP_MPN(addmul_7)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_7(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_addmul_8 __GPGMP_MPN(addmul_8)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_8(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        /* Alternative entry point in gpmpn_addmul_2 for the benefit of gpmpn_sqr_basecase.  */
#define gpmpn_addmul_2s __GPGMP_MPN(addmul_2s)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_2s(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        /* Override gpmpn_addlsh1_n, gpmpn_addlsh2_n, gpmpn_sublsh1_n, etc with gpmpn_addlsh_n,
           etc when !HAVE_NATIVE the former but HAVE_NATIVE_ the latter.  Similarly,
           override foo_ip1 functions with foo.  We then lie and say these macros
           represent native functions, but leave a trace by using the value 2 rather
           than 1.  */

#if HAVE_NATIVE_gpmpn_addlsh_n && !HAVE_NATIVE_gpmpn_addlsh1_n
#define gpmpn_addlsh1_n(a, b, c, d) gpmpn_addlsh_n(a, b, c, d, 1)
#define HAVE_NATIVE_gpmpn_addlsh1_n 2
#endif

#if HAVE_NATIVE_gpmpn_addlsh_nc && !HAVE_NATIVE_gpmpn_addlsh1_nc
#define gpmpn_addlsh1_nc(a, b, c, d, x) gpmpn_addlsh_nc(a, b, c, d, 1, x)
#define HAVE_NATIVE_gpmpn_addlsh1_nc 2
#endif

#if HAVE_NATIVE_gpmpn_addlsh1_n && !HAVE_NATIVE_gpmpn_addlsh1_n_ip1
#define gpmpn_addlsh1_n_ip1(a, b, n) gpmpn_addlsh1_n(a, a, b, n)
#define HAVE_NATIVE_gpmpn_addlsh1_n_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_addlsh1_nc && !HAVE_NATIVE_gpmpn_addlsh1_nc_ip1
#define gpmpn_addlsh1_nc_ip1(a, b, n, c) gpmpn_addlsh1_nc(a, a, b, n, c)
#define HAVE_NATIVE_gpmpn_addlsh1_nc_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_addlsh_n && !HAVE_NATIVE_gpmpn_addlsh2_n
#define gpmpn_addlsh2_n(a, b, c, d) gpmpn_addlsh_n(a, b, c, d, 2)
#define HAVE_NATIVE_gpmpn_addlsh2_n 2
#endif

#if HAVE_NATIVE_gpmpn_addlsh_nc && !HAVE_NATIVE_gpmpn_addlsh2_nc
#define gpmpn_addlsh2_nc(a, b, c, d, x) gpmpn_addlsh_nc(a, b, c, d, 2, x)
#define HAVE_NATIVE_gpmpn_addlsh2_nc 2
#endif

#if HAVE_NATIVE_gpmpn_addlsh2_n && !HAVE_NATIVE_gpmpn_addlsh2_n_ip1
#define gpmpn_addlsh2_n_ip1(a, b, n) gpmpn_addlsh2_n(a, a, b, n)
#define HAVE_NATIVE_gpmpn_addlsh2_n_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_addlsh2_nc && !HAVE_NATIVE_gpmpn_addlsh2_nc_ip1
#define gpmpn_addlsh2_nc_ip1(a, b, n, c) gpmpn_addlsh2_nc(a, a, b, n, c)
#define HAVE_NATIVE_gpmpn_addlsh2_nc_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh_n && !HAVE_NATIVE_gpmpn_sublsh1_n
#define gpmpn_sublsh1_n(a, b, c, d) gpmpn_sublsh_n(a, b, c, d, 1)
#define HAVE_NATIVE_gpmpn_sublsh1_n 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh_nc && !HAVE_NATIVE_gpmpn_sublsh1_nc
#define gpmpn_sublsh1_nc(a, b, c, d, x) gpmpn_sublsh_nc(a, b, c, d, 1, x)
#define HAVE_NATIVE_gpmpn_sublsh1_nc 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh1_n && !HAVE_NATIVE_gpmpn_sublsh1_n_ip1
#define gpmpn_sublsh1_n_ip1(a, b, n) gpmpn_sublsh1_n(a, a, b, n)
#define HAVE_NATIVE_gpmpn_sublsh1_n_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh1_nc && !HAVE_NATIVE_gpmpn_sublsh1_nc_ip1
#define gpmpn_sublsh1_nc_ip1(a, b, n, c) gpmpn_sublsh1_nc(a, a, b, n, c)
#define HAVE_NATIVE_gpmpn_sublsh1_nc_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh_n && !HAVE_NATIVE_gpmpn_sublsh2_n
#define gpmpn_sublsh2_n(a, b, c, d) gpmpn_sublsh_n(a, b, c, d, 2)
#define HAVE_NATIVE_gpmpn_sublsh2_n 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh_nc && !HAVE_NATIVE_gpmpn_sublsh2_nc
#define gpmpn_sublsh2_nc(a, b, c, d, x) gpmpn_sublsh_nc(a, b, c, d, 2, x)
#define HAVE_NATIVE_gpmpn_sublsh2_nc 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh2_n && !HAVE_NATIVE_gpmpn_sublsh2_n_ip1
#define gpmpn_sublsh2_n_ip1(a, b, n) gpmpn_sublsh2_n(a, a, b, n)
#define HAVE_NATIVE_gpmpn_sublsh2_n_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_sublsh2_nc && !HAVE_NATIVE_gpmpn_sublsh2_nc_ip1
#define gpmpn_sublsh2_nc_ip1(a, b, n, c) gpmpn_sublsh2_nc(a, a, b, n, c)
#define HAVE_NATIVE_gpmpn_sublsh2_nc_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh_n && !HAVE_NATIVE_gpmpn_rsblsh1_n
#define gpmpn_rsblsh1_n(a, b, c, d) gpmpn_rsblsh_n(a, b, c, d, 1)
#define HAVE_NATIVE_gpmpn_rsblsh1_n 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh_nc && !HAVE_NATIVE_gpmpn_rsblsh1_nc
#define gpmpn_rsblsh1_nc(a, b, c, d, x) gpmpn_rsblsh_nc(a, b, c, d, 1, x)
#define HAVE_NATIVE_gpmpn_rsblsh1_nc 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh1_n && !HAVE_NATIVE_gpmpn_rsblsh1_n_ip1
#define gpmpn_rsblsh1_n_ip1(a, b, n) gpmpn_rsblsh1_n(a, a, b, n)
#define HAVE_NATIVE_gpmpn_rsblsh1_n_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh1_nc && !HAVE_NATIVE_gpmpn_rsblsh1_nc_ip1
#define gpmpn_rsblsh1_nc_ip1(a, b, n, c) gpmpn_rsblsh1_nc(a, a, b, n, c)
#define HAVE_NATIVE_gpmpn_rsblsh1_nc_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh_n && !HAVE_NATIVE_gpmpn_rsblsh2_n
#define gpmpn_rsblsh2_n(a, b, c, d) gpmpn_rsblsh_n(a, b, c, d, 2)
#define HAVE_NATIVE_gpmpn_rsblsh2_n 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh_nc && !HAVE_NATIVE_gpmpn_rsblsh2_nc
#define gpmpn_rsblsh2_nc(a, b, c, d, x) gpmpn_rsblsh_nc(a, b, c, d, 2, x)
#define HAVE_NATIVE_gpmpn_rsblsh2_nc 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh2_n && !HAVE_NATIVE_gpmpn_rsblsh2_n_ip1
#define gpmpn_rsblsh2_n_ip1(a, b, n) gpmpn_rsblsh2_n(a, a, b, n)
#define HAVE_NATIVE_gpmpn_rsblsh2_n_ip1 2
#endif

#if HAVE_NATIVE_gpmpn_rsblsh2_nc && !HAVE_NATIVE_gpmpn_rsblsh2_nc_ip1
#define gpmpn_rsblsh2_nc_ip1(a, b, n, c) gpmpn_rsblsh2_nc(a, a, b, n, c)
#define HAVE_NATIVE_gpmpn_rsblsh2_nc_ip1 2
#endif

#ifndef gpmpn_addlsh1_n
#define gpmpn_addlsh1_n __GPGMP_MPN(addlsh1_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh1_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_addlsh1_nc
#define gpmpn_addlsh1_nc __GPGMP_MPN(addlsh1_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh1_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif
#ifndef gpmpn_addlsh1_n_ip1
#define gpmpn_addlsh1_n_ip1 __GPGMP_MPN(addlsh1_n_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh1_n_ip1(mp_ptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_addlsh1_nc_ip1
#define gpmpn_addlsh1_nc_ip1 __GPGMP_MPN(addlsh1_nc_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh1_nc_ip1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

#ifndef gpmpn_addlsh2_n
#define gpmpn_addlsh2_n __GPGMP_MPN(addlsh2_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh2_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_addlsh2_nc
#define gpmpn_addlsh2_nc __GPGMP_MPN(addlsh2_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh2_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif
#ifndef gpmpn_addlsh2_n_ip1
#define gpmpn_addlsh2_n_ip1 __GPGMP_MPN(addlsh2_n_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh2_n_ip1(mp_ptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_addlsh2_nc_ip1
#define gpmpn_addlsh2_nc_ip1 __GPGMP_MPN(addlsh2_nc_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh2_nc_ip1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

#ifndef gpmpn_addlsh_n
#define gpmpn_addlsh_n __GPGMP_MPN(addlsh_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
#endif
#ifndef gpmpn_addlsh_nc
#define gpmpn_addlsh_nc __GPGMP_MPN(addlsh_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
#endif
#ifndef gpmpn_addlsh_n_ip1
#define gpmpn_addlsh_n_ip1 __GPGMP_MPN(addlsh_n_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh_n_ip1(mp_ptr, mp_srcptr, mp_size_t, unsigned int);
#endif
#ifndef gpmpn_addlsh_nc_ip1
#define gpmpn_addlsh_nc_ip1 __GPGMP_MPN(addlsh_nc_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addlsh_nc_ip1(mp_ptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
#endif

#ifndef gpmpn_sublsh1_n
#define gpmpn_sublsh1_n __GPGMP_MPN(sublsh1_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh1_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_sublsh1_nc
#define gpmpn_sublsh1_nc __GPGMP_MPN(sublsh1_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh1_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif
#ifndef gpmpn_sublsh1_n_ip1
#define gpmpn_sublsh1_n_ip1 __GPGMP_MPN(sublsh1_n_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh1_n_ip1(mp_ptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_sublsh1_nc_ip1
#define gpmpn_sublsh1_nc_ip1 __GPGMP_MPN(sublsh1_nc_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh1_nc_ip1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

#ifndef gpmpn_sublsh2_n
#define gpmpn_sublsh2_n __GPGMP_MPN(sublsh2_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh2_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_sublsh2_nc
#define gpmpn_sublsh2_nc __GPGMP_MPN(sublsh2_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh2_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif
#ifndef gpmpn_sublsh2_n_ip1
#define gpmpn_sublsh2_n_ip1 __GPGMP_MPN(sublsh2_n_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh2_n_ip1(mp_ptr, mp_srcptr, mp_size_t);
#endif
#ifndef gpmpn_sublsh2_nc_ip1
#define gpmpn_sublsh2_nc_ip1 __GPGMP_MPN(sublsh2_nc_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh2_nc_ip1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

#ifndef gpmpn_sublsh_n
#define gpmpn_sublsh_n __GPGMP_MPN(sublsh_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
#endif
#ifndef gpmpn_sublsh_nc
#define gpmpn_sublsh_nc __GPGMP_MPN(sublsh_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
#endif
#ifndef gpmpn_sublsh_n_ip1
#define gpmpn_sublsh_n_ip1 __GPGMP_MPN(sublsh_n_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh_n_ip1(mp_ptr, mp_srcptr, mp_size_t, unsigned int);
#endif
#ifndef gpmpn_sublsh_nc_ip1
#define gpmpn_sublsh_nc_ip1 __GPGMP_MPN(sublsh_nc_ip1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sublsh_nc_ip1(mp_ptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
#endif

#define gpmpn_rsblsh1_n __GPGMP_MPN(rsblsh1_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_signed_t gpmpn_rsblsh1_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_rsblsh1_nc __GPGMP_MPN(rsblsh1_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_signed_t gpmpn_rsblsh1_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_rsblsh2_n __GPGMP_MPN(rsblsh2_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_signed_t gpmpn_rsblsh2_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_rsblsh2_nc __GPGMP_MPN(rsblsh2_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_signed_t gpmpn_rsblsh2_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_rsblsh_n __GPGMP_MPN(rsblsh_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_signed_t gpmpn_rsblsh_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
#define gpmpn_rsblsh_nc __GPGMP_MPN(rsblsh_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_signed_t gpmpn_rsblsh_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);

#define gpmpn_rsh1add_n __GPGMP_MPN(rsh1add_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_rsh1add_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_rsh1add_nc __GPGMP_MPN(rsh1add_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_rsh1add_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_rsh1sub_n __GPGMP_MPN(rsh1sub_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_rsh1sub_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define gpmpn_rsh1sub_nc __GPGMP_MPN(rsh1sub_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_rsh1sub_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#ifndef gpmpn_lshiftc /* if not done with cpuvec in a fat binary */
#define gpmpn_lshiftc __GPGMP_MPN(lshiftc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_lshiftc(mp_ptr, mp_srcptr, mp_size_t, unsigned int);
#endif

#define gpmpn_add_err1_n __GPGMP_MPN(add_err1_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_err1_n(mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_add_err2_n __GPGMP_MPN(add_err2_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_err2_n(mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_add_err3_n __GPGMP_MPN(add_err3_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_err3_n(mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_sub_err1_n __GPGMP_MPN(sub_err1_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_err1_n(mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_sub_err2_n __GPGMP_MPN(sub_err2_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_err2_n(mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_sub_err3_n __GPGMP_MPN(sub_err3_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_err3_n(mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_add_n_sub_n __GPGMP_MPN(add_n_sub_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_n_sub_n(mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_add_n_sub_nc __GPGMP_MPN(add_n_sub_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_n_sub_nc(mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_addaddmul_1msb0 __GPGMP_MPN(addaddmul_1msb0)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addaddmul_1msb0(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#define gpmpn_divrem_1c __GPGMP_MPN(divrem_1c)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divrem_1c(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#define gpmpn_dump __GPGMP_MPN(dump)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_dump(mp_srcptr, mp_size_t);

#define gpmpn_fib2_ui __GPGMP_MPN(fib2_ui)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_fib2_ui(mp_ptr, mp_ptr, unsigned long);

#define gpmpn_fib2m __GPGMP_MPN(fib2m)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_fib2m(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_strongfibo __GPGMP_MPN(strongfibo)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_strongfibo(mp_srcptr, mp_size_t, mp_ptr);

        /* Remap names of internal mpn functions.  */
#define __clz_tab __GPGMP_MPN(clz_tab)
#define gpmpn_udiv_w_sdiv __GPGMP_MPN(udiv_w_sdiv)

#define gpmpn_jacobi_base __GPGMP_MPN(jacobi_base)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_jacobi_base(mp_limb_t, mp_limb_t, int) ATTRIBUTE_CONST;

#define gpmpn_jacobi_2 __GPGMP_MPN(jacobi_2)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_jacobi_2(mp_srcptr, mp_srcptr, unsigned);

#define gpmpn_jacobi_n __GPGMP_MPN(jacobi_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_jacobi_n(mp_ptr, mp_ptr, mp_size_t, unsigned);

#define gpmpn_mod_1c __GPGMP_MPN(mod_1c)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_1c(mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_mul_1c __GPGMP_MPN(mul_1c)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_1c(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#define gpmpn_mul_2 __GPGMP_MPN(mul_2)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_2(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_mul_3 __GPGMP_MPN(mul_3)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_3(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_mul_4 __GPGMP_MPN(mul_4)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_4(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_mul_5 __GPGMP_MPN(mul_5)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_5(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define gpmpn_mul_6 __GPGMP_MPN(mul_6)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_6(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#ifndef gpmpn_mul_basecase /* if not done with cpuvec in a fat binary */
#define gpmpn_mul_basecase __GPGMP_MPN(mul_basecase)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mul_basecase(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#endif

#define gpmpn_mullo_n __GPGMP_MPN(mullo_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mullo_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#ifndef gpmpn_mullo_basecase /* if not done with cpuvec in a fat binary */
#define gpmpn_mullo_basecase __GPGMP_MPN(mullo_basecase)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mullo_basecase(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif

#ifndef gpmpn_sqr_basecase /* if not done with cpuvec in a fat binary */
#define gpmpn_sqr_basecase __GPGMP_MPN(sqr_basecase)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqr_basecase(mp_ptr, mp_srcptr, mp_size_t);
#endif

#define gpmpn_sqrlo __GPGMP_MPN(sqrlo)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqrlo(mp_ptr, mp_srcptr, mp_size_t);

#define gpmpn_sqrlo_basecase __GPGMP_MPN(sqrlo_basecase)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqrlo_basecase(mp_ptr, mp_srcptr, mp_size_t);

#define gpmpn_mulmid_basecase __GPGMP_MPN(mulmid_basecase)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mulmid_basecase(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_mulmid_n __GPGMP_MPN(mulmid_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mulmid_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_mulmid __GPGMP_MPN(mulmid)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mulmid(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_submul_1c __GPGMP_MPN(submul_1c)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_submul_1c(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#ifndef gpmpn_redc_1 /* if not done with cpuvec in a fat binary */
#define gpmpn_redc_1 __GPGMP_MPN(redc_1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_redc_1(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

#ifndef gpmpn_redc_2 /* if not done with cpuvec in a fat binary */
#define gpmpn_redc_2 __GPGMP_MPN(redc_2)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_redc_2(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
#endif

#define gpmpn_redc_n __GPGMP_MPN(redc_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_redc_n(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#ifndef gpmpn_mod_1_1p_cps /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1_1p_cps __GPGMP_MPN(mod_1_1p_cps)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mod_1_1p_cps(mp_limb_t[4], mp_limb_t);
#endif
#ifndef gpmpn_mod_1_1p /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1_1p __GPGMP_MPN(mod_1_1p)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_1_1p(mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t[4]) __GMP_ATTRIBUTE_PURE;
#endif

#ifndef gpmpn_mod_1s_2p_cps /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1s_2p_cps __GPGMP_MPN(mod_1s_2p_cps)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mod_1s_2p_cps(mp_limb_t[5], mp_limb_t);
#endif
#ifndef gpmpn_mod_1s_2p /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1s_2p __GPGMP_MPN(mod_1s_2p)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_1s_2p(mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t[5]) __GMP_ATTRIBUTE_PURE;
#endif

#ifndef gpmpn_mod_1s_3p_cps /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1s_3p_cps __GPGMP_MPN(mod_1s_3p_cps)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mod_1s_3p_cps(mp_limb_t[6], mp_limb_t);
#endif
#ifndef gpmpn_mod_1s_3p /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1s_3p __GPGMP_MPN(mod_1s_3p)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_1s_3p(mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t[6]) __GMP_ATTRIBUTE_PURE;
#endif

#ifndef gpmpn_mod_1s_4p_cps /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1s_4p_cps __GPGMP_MPN(mod_1s_4p_cps)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mod_1s_4p_cps(mp_limb_t[7], mp_limb_t);
#endif
#ifndef gpmpn_mod_1s_4p /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_1s_4p __GPGMP_MPN(mod_1s_4p)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_1s_4p(mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t[7]) __GMP_ATTRIBUTE_PURE;
#endif

#define gpmpn_bc_mulmod_bnm1 __GPGMP_MPN(bc_mulmod_bnm1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_bc_mulmod_bnm1(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_mulmod_bnm1 __GPGMP_MPN(mulmod_bnm1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mulmod_bnm1(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_mulmod_bnm1_next_size __GPGMP_MPN(mulmod_bnm1_next_size)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_mulmod_bnm1_next_size(mp_size_t) ATTRIBUTE_CONST;
        ANYCALLER static inline mp_size_t gpmpn_mulmod_bnm1_itch(mp_size_t rn, mp_size_t an, mp_size_t bn)
        {
            mp_size_t n, itch;
            n = rn >> 1;
            itch = rn + 4 +
                   (an > n ? (bn > n ? rn : n) : 0);
            return itch;
        }
        GPGMP_MPN_NAMESPACE_END
#ifndef MOD_BKNP1_USE11
#define MOD_BKNP1_USE11 ((GMP_NUMB_BITS % 8 != 0) && (GMP_NUMB_BITS % 2 == 0))
#endif
#ifndef MOD_BKNP1_ONLY3
#define MOD_BKNP1_ONLY3 0
#endif
        GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_mulmod_bknp1 __GPGMP_MPN(mulmod_bknp1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mulmod_bknp1(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned, mp_ptr);
        ANYCALLER static inline mp_size_t
        gpmpn_mulmod_bknp1_itch(mp_size_t rn)
        {
            return rn << 2;
        }
        GPGMP_MPN_NAMESPACE_END
#if MOD_BKNP1_ONLY3
#define MPN_MULMOD_BKNP1_USABLE(rn, k, mn)                      \
        ((GMP_NUMB_BITS % 8 == 0) && ((mn) >= 18) && ((rn) > 16) && \
         (((rn) % ((k) = 3) == 0)))
#else
#define MPN_MULMOD_BKNP1_USABLE(rn, k, mn)                                   \
        (((GMP_NUMB_BITS % 8 == 0) && ((mn) >= 18) && ((rn) > 16) &&             \
          (((rn) % ((k) = 3) == 0) ||                                            \
           (((GMP_NUMB_BITS % 16 != 0) || (((mn) >= 35) && ((rn) >= 32))) &&     \
            (((GMP_NUMB_BITS % 16 == 0) && ((rn) % ((k) = 5) == 0)) ||           \
             (((mn) >= 49) &&                                                    \
              (((rn) % ((k) = 7) == 0) ||                                        \
               ((GMP_NUMB_BITS % 16 == 0) && ((mn) >= 104) && ((rn) >= 64) &&    \
                ((MOD_BKNP1_USE11 && ((rn) % ((k) = 11) == 0)) ||                \
                 ((rn) % ((k) = 13) == 0) ||                                     \
                 ((GMP_NUMB_BITS % 32 == 0) && ((mn) >= 136) && ((rn) >= 128) && \
                  ((rn) % ((k) = 17) == 0)))))))))) ||                           \
         ((GMP_NUMB_BITS % 16 != 0) && MOD_BKNP1_USE11 &&                        \
          ((mn) >= 104) && ((rn) >= 64) && ((rn) % ((k) = 11) == 0)))
#endif
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_sqrmod_bknp1 __GPGMP_MPN(sqrmod_bknp1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqrmod_bknp1(mp_ptr, mp_srcptr, mp_size_t, unsigned, mp_ptr);
        ANYCALLER static inline mp_size_t
        gpmpn_sqrmod_bknp1_itch(mp_size_t rn)
        {
            return rn * 3;
        }
#if MOD_BKNP1_ONLY3
#define MPN_SQRMOD_BKNP1_USABLE(rn, k, mn) \
        MPN_MULMOD_BKNP1_USABLE(rn, k, mn)
#else
#define MPN_SQRMOD_BKNP1_USABLE(rn, k, mn)                                   \
        (((GMP_NUMB_BITS % 8 == 0) && ((mn) >= 27) && ((rn) > 24) &&             \
          (((rn) % ((k) = 3) == 0) ||                                            \
           (((GMP_NUMB_BITS % 16 != 0) || (((mn) >= 55) && ((rn) > 50))) &&      \
            (((GMP_NUMB_BITS % 16 == 0) && ((rn) % ((k) = 5) == 0)) ||           \
             (((mn) >= 56) &&                                                    \
              (((rn) % ((k) = 7) == 0) ||                                        \
               ((GMP_NUMB_BITS % 16 == 0) && ((mn) >= 143) && ((rn) >= 128) &&   \
                ((MOD_BKNP1_USE11 && ((rn) % ((k) = 11) == 0)) ||                \
                 ((rn) % ((k) = 13) == 0) ||                                     \
                 ((GMP_NUMB_BITS % 32 == 0) && ((mn) >= 272) && ((rn) >= 256) && \
                  ((rn) % ((k) = 17) == 0)))))))))) ||                           \
         ((GMP_NUMB_BITS % 16 != 0) && MOD_BKNP1_USE11 &&                        \
          ((mn) >= 143) && ((rn) >= 128) && ((rn) % ((k) = 11) == 0)))
#endif

#define gpmpn_sqrmod_bnm1 __GPGMP_MPN(sqrmod_bnm1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqrmod_bnm1(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_sqrmod_bnm1_next_size __GPGMP_MPN(sqrmod_bnm1_next_size)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sqrmod_bnm1_next_size(mp_size_t) ATTRIBUTE_CONST;
        ANYCALLER static inline mp_size_t gpmpn_sqrmod_bnm1_itch(mp_size_t rn, mp_size_t an)
        {
            mp_size_t n, itch;
            n = rn >> 1;
            itch = rn + 3 +
                   (an > n ? an : 0);
            return itch;
        }
GPGMP_MPN_NAMESPACE_END
        /* Pseudo-random number generator function pointers structure.  */
        typedef struct
        {
            void (*randseed_fn)(gmp_randstate_ptr, mpz_srcptr);
            void (*randget_fn)(gmp_randstate_ptr, mp_ptr, unsigned long int);
            void (*randclear_fn)(gmp_randstate_ptr);
            void (*randiset_fn)(gmp_randstate_ptr, gmp_randstate_srcptr);
        } gmp_randfnptr_t;

        /* Macro to obtain a void pointer to the function pointers structure.  */
#define RNG_FNPTR(rstate) ((rstate)->_mp_algdata._mp_lc)

        /* Macro to obtain a pointer to the generator's state.
           When used as a lvalue the rvalue needs to be cast to mp_ptr.  */
#define RNG_STATE(rstate) ((rstate)->_mp_seed->_mp_d)

        /* Write a given number of random bits to rp.  */
#define _gmp_rand(rp, state, bits)                                                   \
        do                                                                               \
        {                                                                                \
            gmp_randstate_ptr __rstate = (state);                                        \
            (*((gmp_randfnptr_t *)RNG_FNPTR(__rstate))->randget_fn)(__rstate, rp, bits); \
        } while (0)
GPGMP_MPN_NAMESPACE_BEGIN
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gmp_randinit_mt_noseed(gmp_randstate_ptr);
GPGMP_MPN_NAMESPACE_END
        /* __gpgmp_rands is the global state for the old-style random functions, and
           is also used in the test programs (hence the __GPGMP_DECLSPEC __GPGMP_CALLERTYPE).

           There's no seeding here, so mpz_random etc will generate the same
           sequence every time.  This is not unlike the C library random functions
           if you don't seed them, so perhaps it's acceptable.  Digging up a seed
           from /dev/random or the like would work on many systems, but might
           encourage a false confidence, since it'd be pretty much impossible to do
           something that would work reliably everywhere.  In any case the new style
           functions are recommended to applications which care about randomness, so
           the old functions aren't too important.  */

        __GPGMP_DECLSPEC extern char __gpgmp_rands_initialized;
        __GPGMP_DECLSPEC extern gmp_randstate_t __gpgmp_rands;

#define RANDS                                                                \
        ((__gpgmp_rands_initialized ? 0                                            \
                                  : (__gpgmp_rands_initialized = 1,                \
                                     __gmp_randinit_mt_noseed(__gpgmp_rands), 0)), \
         __gpgmp_rands)

        /* this is used by the test programs, to free memory */
#define RANDS_CLEAR()                    \
        do                                   \
        {                                    \
            if (__gpgmp_rands_initialized)     \
            {                                \
                __gpgmp_rands_initialized = 0; \
                gmp_randclear(__gpgmp_rands);  \
            }                                \
        } while (0)

        /* For a threshold between algorithms A and B, size>=thresh is where B
           should be used.  Special value MP_SIZE_T_MAX means only ever use A, or
           value 0 means only ever use B.  The tests for these special values will
           be compile-time constants, so the compiler should be able to eliminate
           the code for the unwanted algorithm.  */

#if !defined(__GNUC__) || __GNUC__ < 2
#define ABOVE_THRESHOLD(size, thresh) \
        ((thresh) == 0 || ((thresh) != MP_SIZE_T_MAX && (size) >= (thresh)))
#else
#define ABOVE_THRESHOLD(size, thresh) \
        ((__builtin_constant_p(thresh) && (thresh) == 0) || (!(__builtin_constant_p(thresh) && (thresh) == MP_SIZE_T_MAX) && (size) >= (thresh)))
#endif
#define BELOW_THRESHOLD(size, thresh) (!ABOVE_THRESHOLD(size, thresh))

        /* The minimal supported value for Toom22 depends also on Toom32 and
           Toom42 implementations. */
#define MPN_TOOM22_MUL_MINSIZE 6
#define MPN_TOOM2_SQR_MINSIZE 4

#define MPN_TOOM33_MUL_MINSIZE 17
#define MPN_TOOM3_SQR_MINSIZE 17

#define MPN_TOOM44_MUL_MINSIZE 30
#define MPN_TOOM4_SQR_MINSIZE 30

#define MPN_TOOM6H_MUL_MINSIZE 46
#define MPN_TOOM6_SQR_MINSIZE 46

#define MPN_TOOM8H_MUL_MINSIZE 86
#define MPN_TOOM8_SQR_MINSIZE 86

#define MPN_TOOM32_MUL_MINSIZE 10
#define MPN_TOOM42_MUL_MINSIZE 10
#define MPN_TOOM43_MUL_MINSIZE 25
#define MPN_TOOM53_MUL_MINSIZE 17
#define MPN_TOOM54_MUL_MINSIZE 31
#define MPN_TOOM63_MUL_MINSIZE 49

#define MPN_TOOM42_MULMID_MINSIZE 4
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_sqr_diagonal __GPGMP_MPN(sqr_diagonal)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqr_diagonal(mp_ptr, mp_srcptr, mp_size_t);

#define gpmpn_sqr_diag_addlsh1 __GPGMP_MPN(sqr_diag_addlsh1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqr_diag_addlsh1(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define gpmpn_toom_interpolate_5pts __GPGMP_MPN(toom_interpolate_5pts)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom_interpolate_5pts(mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_limb_t);
GPGMP_MPN_NAMESPACE_END
        enum toom6_flags
        {
            toom6_all_pos = 0,
            toom6_vm1_neg = 1,
            toom6_vm2_neg = 2
        };
        GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_toom_interpolate_6pts __GPGMP_MPN(toom_interpolate_6pts)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom_interpolate_6pts(mp_ptr, mp_size_t, enum toom6_flags, mp_ptr, mp_ptr, mp_ptr, mp_size_t);
GPGMP_MPN_NAMESPACE_END
        enum toom7_flags
        {
            toom7_w1_neg = 1,
            toom7_w3_neg = 2
        };
        GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_toom_interpolate_7pts __GPGMP_MPN(toom_interpolate_7pts)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom_interpolate_7pts(mp_ptr, mp_size_t, enum toom7_flags, mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

#define gpmpn_toom_interpolate_8pts __GPGMP_MPN(toom_interpolate_8pts)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom_interpolate_8pts(mp_ptr, mp_size_t, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

#define gpmpn_toom_interpolate_12pts __GPGMP_MPN(toom_interpolate_12pts)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom_interpolate_12pts(mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_ptr);

#define gpmpn_toom_interpolate_16pts __GPGMP_MPN(toom_interpolate_16pts)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom_interpolate_16pts(mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_ptr);

#define gpmpn_toom_couple_handling __GPGMP_MPN(toom_couple_handling)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom_couple_handling(mp_ptr, mp_size_t, mp_ptr, int, mp_size_t, int, int);

#define gpmpn_toom_eval_dgr3_pm1 __GPGMP_MPN(toom_eval_dgr3_pm1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_toom_eval_dgr3_pm1(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define gpmpn_toom_eval_dgr3_pm2 __GPGMP_MPN(toom_eval_dgr3_pm2)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_toom_eval_dgr3_pm2(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define gpmpn_toom_eval_pm1 __GPGMP_MPN(toom_eval_pm1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_toom_eval_pm1(mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define gpmpn_toom_eval_pm2 __GPGMP_MPN(toom_eval_pm2)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_toom_eval_pm2(mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define gpmpn_toom_eval_pm2exp __GPGMP_MPN(toom_eval_pm2exp)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_toom_eval_pm2exp(mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, unsigned, mp_ptr);

#define gpmpn_toom_eval_pm2rexp __GPGMP_MPN(toom_eval_pm2rexp)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_toom_eval_pm2rexp(mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, unsigned, mp_ptr);

#define gpmpn_toom22_mul __GPGMP_MPN(toom22_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom22_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom32_mul __GPGMP_MPN(toom32_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom32_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom42_mul __GPGMP_MPN(toom42_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom42_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom52_mul __GPGMP_MPN(toom52_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom52_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom62_mul __GPGMP_MPN(toom62_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom62_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom2_sqr __GPGMP_MPN(toom2_sqr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom2_sqr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom33_mul __GPGMP_MPN(toom33_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom33_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom43_mul __GPGMP_MPN(toom43_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom43_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom53_mul __GPGMP_MPN(toom53_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom53_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom54_mul __GPGMP_MPN(toom54_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom54_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom63_mul __GPGMP_MPN(toom63_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom63_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom3_sqr __GPGMP_MPN(toom3_sqr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom3_sqr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom44_mul __GPGMP_MPN(toom44_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom44_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom4_sqr __GPGMP_MPN(toom4_sqr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom4_sqr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom6h_mul __GPGMP_MPN(toom6h_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom6h_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom6_sqr __GPGMP_MPN(toom6_sqr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom6_sqr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom8h_mul __GPGMP_MPN(toom8h_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom8h_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom8_sqr __GPGMP_MPN(toom8_sqr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom8_sqr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_toom42_mulmid __GPGMP_MPN(toom42_mulmid)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_toom42_mulmid(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_fft_best_k __GPGMP_MPN(fft_best_k)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_fft_best_k(mp_size_t, int) ATTRIBUTE_CONST;

#define gpmpn_mul_fft __GPGMP_MPN(mul_fft)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_fft(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, int);

#define gpmpn_mul_fft_full __GPGMP_MPN(mul_fft_full)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mul_fft_full(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_nussbaumer_mul __GPGMP_MPN(nussbaumer_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_nussbaumer_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define gpmpn_fft_next_size __GPGMP_MPN(fft_next_size)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_fft_next_size(mp_size_t, int) ATTRIBUTE_CONST;

#define gpmpn_div_qr_1n_pi1 __GPGMP_MPN(div_qr_1n_pi1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_div_qr_1n_pi1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

#define gpmpn_div_qr_2n_pi1 __GPGMP_MPN(div_qr_2n_pi1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_div_qr_2n_pi1(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

#define gpmpn_div_qr_2u_pi1 __GPGMP_MPN(div_qr_2u_pi1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_div_qr_2u_pi1(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int, mp_limb_t);

#define gpmpn_sbpi1_div_qr __GPGMP_MPN(sbpi1_div_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sbpi1_div_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_sbpi1_div_q __GPGMP_MPN(sbpi1_div_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sbpi1_div_q(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_sbpi1_divappr_q __GPGMP_MPN(sbpi1_divappr_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sbpi1_divappr_q(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_dcpi1_div_qr __GPGMP_MPN(dcpi1_div_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_dcpi1_div_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);
#define gpmpn_dcpi1_div_qr_n __GPGMP_MPN(dcpi1_div_qr_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_dcpi1_div_qr_n(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, gmp_pi1_t *, mp_ptr);

#define gpmpn_dcpi1_div_q __GPGMP_MPN(dcpi1_div_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_dcpi1_div_q(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);

#define gpmpn_dcpi1_divappr_q __GPGMP_MPN(dcpi1_divappr_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_dcpi1_divappr_q(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);

#define gpmpn_mu_div_qr __GPGMP_MPN(mu_div_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mu_div_qr(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_mu_div_qr_itch __GPGMP_MPN(mu_div_qr_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_mu_div_qr_itch(mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;

#define gpmpn_preinv_mu_div_qr __GPGMP_MPN(preinv_mu_div_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_preinv_mu_div_qr(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_preinv_mu_div_qr_itch __GPGMP_MPN(preinv_mu_div_qr_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_preinv_mu_div_qr_itch(mp_size_t, mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_mu_divappr_q __GPGMP_MPN(mu_divappr_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mu_divappr_q(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_mu_divappr_q_itch __GPGMP_MPN(mu_divappr_q_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_mu_divappr_q_itch(mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;

#define gpmpn_mu_div_q __GPGMP_MPN(mu_div_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mu_div_q(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_mu_div_q_itch __GPGMP_MPN(mu_div_q_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_mu_div_q_itch(mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;

#define gpmpn_div_q __GPGMP_MPN(div_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_div_q(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define gpmpn_invert __GPGMP_MPN(invert)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_invert(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_invert_itch(n) gpmpn_invertappr_itch(n)

#define gpmpn_ni_invertappr __GPGMP_MPN(ni_invertappr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_ni_invertappr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_invertappr __GPGMP_MPN(invertappr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_invertappr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_invertappr_itch(n) (2 * (n))

#define gpmpn_binvert __GPGMP_MPN(binvert)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_binvert(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_binvert_itch __GPGMP_MPN(binvert_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_binvert_itch(mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_bdiv_q_1 __GPGMP_MPN(bdiv_q_1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_bdiv_q_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_pi1_bdiv_q_1 __GPGMP_MPN(pi1_bdiv_q_1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_pi1_bdiv_q_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int);

#define gpmpn_sbpi1_bdiv_qr __GPGMP_MPN(sbpi1_bdiv_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sbpi1_bdiv_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_sbpi1_bdiv_q __GPGMP_MPN(sbpi1_bdiv_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sbpi1_bdiv_q(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_sbpi1_bdiv_r __GPGMP_MPN(sbpi1_bdiv_r)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sbpi1_bdiv_r(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_dcpi1_bdiv_qr __GPGMP_MPN(dcpi1_bdiv_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_dcpi1_bdiv_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
#define gpmpn_dcpi1_bdiv_qr_n_itch __GPGMP_MPN(dcpi1_bdiv_qr_n_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_dcpi1_bdiv_qr_n_itch(mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_dcpi1_bdiv_qr_n __GPGMP_MPN(dcpi1_bdiv_qr_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_dcpi1_bdiv_qr_n(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
#define gpmpn_dcpi1_bdiv_q __GPGMP_MPN(dcpi1_bdiv_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_dcpi1_bdiv_q(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_mu_bdiv_qr __GPGMP_MPN(mu_bdiv_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mu_bdiv_qr(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_mu_bdiv_qr_itch __GPGMP_MPN(mu_bdiv_qr_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_mu_bdiv_qr_itch(mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_mu_bdiv_q __GPGMP_MPN(mu_bdiv_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mu_bdiv_q(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_mu_bdiv_q_itch __GPGMP_MPN(mu_bdiv_q_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_mu_bdiv_q_itch(mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_bdiv_qr __GPGMP_MPN(bdiv_qr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_bdiv_qr(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_bdiv_qr_itch __GPGMP_MPN(bdiv_qr_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_bdiv_qr_itch(mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_bdiv_q __GPGMP_MPN(bdiv_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_bdiv_q(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_bdiv_q_itch __GPGMP_MPN(bdiv_q_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_bdiv_q_itch(mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_divexact __GPGMP_MPN(divexact)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_divexact(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#define gpmpn_divexact_itch __GPGMP_MPN(divexact_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_divexact_itch(mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#ifndef gpmpn_bdiv_dbm1c /* if not done with cpuvec in a fat binary */
#define gpmpn_bdiv_dbm1c __GPGMP_MPN(bdiv_dbm1c)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_bdiv_dbm1c(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
#endif

#define gpmpn_bdiv_dbm1(dst, src, size, divisor) \
        gpmpn_bdiv_dbm1c(dst, src, size, divisor, __GMP_CAST(mp_limb_t, 0))

#define gpmpn_powm __GPGMP_MPN(powm)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_powm(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_powlo __GPGMP_MPN(powlo)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_powlo(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define gpmpn_sec_pi1_div_qr __GPGMP_MPN(sec_pi1_div_qr)
#define gpmpn_sec_pi1_div_r __GPGMP_MPN(sec_pi1_div_r)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sec_pi1_div_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
#define gpmpn_sec_pi1_div_r __GPGMP_MPN(sec_pi1_div_r)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_pi1_div_r(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);

GPGMP_MPN_NAMESPACE_END

#ifndef DIVEXACT_BY3_METHOD
#if GMP_NUMB_BITS % 2 == 0 && !defined(HAVE_NATIVE_gpmpn_divexact_by3c)
#define DIVEXACT_BY3_METHOD 0 /* default to using gpmpn_bdiv_dbm1c */
#else
#define DIVEXACT_BY3_METHOD 1
#endif
#endif

#if DIVEXACT_BY3_METHOD == 0
#undef gpmpn_divexact_by3
#define gpmpn_divexact_by3(dst, src, size) \
        (3 & gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 3)))
        /* override gpmpn_divexact_by3c defined in gmp.h */
        /*
        #undef gpmpn_divexact_by3c
        #define gpmpn_divexact_by3c(dst,src,size,cy) \
          (3 & gpmpn_bdiv_dbm1c (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 3, GMP_NUMB_MASK / 3 * cy)))
        */
#endif

#if GMP_NUMB_BITS % 4 == 0
#define gpmpn_divexact_by5(dst, src, size) \
        (7 & 3 * gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 5)))
#endif

#if GMP_NUMB_BITS % 3 == 0
#define gpmpn_divexact_by7(dst, src, size) \
        (7 & 1 * gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 7)))
#endif

#if GMP_NUMB_BITS % 6 == 0
#define gpmpn_divexact_by9(dst, src, size) \
        (15 & 7 * gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 9)))
#endif

#if GMP_NUMB_BITS % 10 == 0
#define gpmpn_divexact_by11(dst, src, size) \
        (15 & 5 * gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 11)))
#endif

#if GMP_NUMB_BITS % 12 == 0
#define gpmpn_divexact_by13(dst, src, size) \
        (15 & 3 * gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 13)))
#endif

#if GMP_NUMB_BITS % 4 == 0
#define gpmpn_divexact_by15(dst, src, size) \
        (15 & 1 * gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 15)))
#endif

#if GMP_NUMB_BITS % 8 == 0
#define gpmpn_divexact_by17(dst, src, size) \
        (31 & 15 * gpmpn_bdiv_dbm1(dst, src, size, __GMP_CAST(mp_limb_t, GMP_NUMB_MASK / 17)))
#endif
GPGMP_MPN_NAMESPACE_BEGIN
#define mpz_divexact_gcd __gmpz_divexact_gcd
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void mpz_divexact_gcd(mpz_ptr, mpz_srcptr, mpz_srcptr);

#define mpz_prodlimbs __gmpz_prodlimbs
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t mpz_prodlimbs(mpz_ptr, mp_ptr, mp_size_t);

#define mpz_oddfac_1 __gmpz_oddfac_1
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void mpz_oddfac_1(mpz_ptr, mp_limb_t, unsigned);

#define mpz_stronglucas __gmpz_stronglucas
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int mpz_stronglucas(mpz_srcptr, mpz_ptr, mpz_ptr);

#define mpz_lucas_mod __gmpz_lucas_mod
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int mpz_lucas_mod(mpz_ptr, mpz_ptr, long, mp_bitcnt_t, mpz_srcptr, mpz_ptr, mpz_ptr);

#define mpz_inp_str_nowhite __gmpz_inp_str_nowhite
#ifdef _GMP_H_HAVE_FILE
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t mpz_inp_str_nowhite(mpz_ptr, FILE *, int, int, size_t);
#endif

#define gpmpn_divisible_p __GPGMP_MPN(divisible_p)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_divisible_p(mp_srcptr, mp_size_t, mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;

#define gpmpn_rootrem __GPGMP_MPN(rootrem)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_rootrem(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_broot __GPGMP_MPN(broot)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_broot(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_broot_invm1 __GPGMP_MPN(broot_invm1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_broot_invm1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_brootinv __GPGMP_MPN(brootinv)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_brootinv(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);

#define gpmpn_bsqrt __GPGMP_MPN(bsqrt)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_bsqrt(mp_ptr, mp_srcptr, mp_bitcnt_t, mp_ptr);

#define gpmpn_bsqrtinv __GPGMP_MPN(bsqrtinv)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_bsqrtinv(mp_ptr, mp_srcptr, mp_bitcnt_t, mp_ptr);
GPGMP_MPN_NAMESPACE_END
#if defined(_CRAY)
#define MPN_COPY_INCR(dst, src, n)                         \
        do                                                     \
        {                                                      \
            int __i; /* Faster on some Crays with plain int */ \
            _Pragma("_CRI ivdep");                             \
            for (__i = 0; __i < (n); __i++)                    \
                (dst)[__i] = (src)[__i];                       \
        } while (0)
#endif
GPGMP_MPN_NAMESPACE_BEGIN
        /* used by test programs, hence __GPGMP_DECLSPEC __GPGMP_CALLERTYPE */
#ifndef gpmpn_copyi /* if not done with cpuvec in a fat binary */
#define gpmpn_copyi __GPGMP_MPN(copyi)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_copyi(mp_ptr, mp_srcptr, mp_size_t);
#endif
GPGMP_MPN_NAMESPACE_END
#if !defined(MPN_COPY_INCR) && HAVE_NATIVE_gpmpn_copyi
#define MPN_COPY_INCR(dst, src, size)               \
        do                                              \
        {                                               \
            ASSERT((size) >= 0);                        \
            ASSERT(MPN_SAME_OR_INCR_P(dst, src, size)); \
            gpmpn_copyi(dst, src, size);                  \
        } while (0)
#endif

        /* Copy N limbs from SRC to DST incrementing, N==0 allowed.  */
#if !defined(MPN_COPY_INCR)
#define MPN_COPY_INCR(dst, src, n)               \
        do                                           \
        {                                            \
            ASSERT((n) >= 0);                        \
            ASSERT(MPN_SAME_OR_INCR_P(dst, src, n)); \
            if ((n) != 0)                            \
            {                                        \
                mp_size_t __n = (n) - 1;             \
                mp_ptr __dst = (dst);                \
                mp_srcptr __src = (src);             \
                mp_limb_t __x;                       \
                __x = *__src++;                      \
                if (__n != 0)                        \
                {                                    \
                    do                               \
                    {                                \
                        *__dst++ = __x;              \
                        __x = *__src++;              \
                    } while (--__n);                 \
                }                                    \
                *__dst++ = __x;                      \
            }                                        \
        } while (0)
#endif

#if defined(_CRAY)
#define MPN_COPY_DECR(dst, src, n)                         \
        do                                                     \
        {                                                      \
            int __i; /* Faster on some Crays with plain int */ \
            _Pragma("_CRI ivdep");                             \
            for (__i = (n) - 1; __i >= 0; __i--)               \
                (dst)[__i] = (src)[__i];                       \
        } while (0)
#endif
GPGMP_MPN_NAMESPACE_BEGIN
        /* used by test programs, hence __GPGMP_DECLSPEC __GPGMP_CALLERTYPE */
#ifndef gpmpn_copyd /* if not done with cpuvec in a fat binary */
#define gpmpn_copyd __GPGMP_MPN(copyd)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_copyd(mp_ptr, mp_srcptr, mp_size_t);
#endif
GPGMP_MPN_NAMESPACE_END
#if !defined(MPN_COPY_DECR) && HAVE_NATIVE_gpmpn_copyd
#define MPN_COPY_DECR(dst, src, size)               \
        do                                              \
        {                                               \
            ASSERT((size) >= 0);                        \
            ASSERT(MPN_SAME_OR_DECR_P(dst, src, size)); \
            gpmpn_copyd(dst, src, size);                  \
        } while (0)
#endif

        /* Copy N limbs from SRC to DST decrementing, N==0 allowed.  */
#if !defined(MPN_COPY_DECR)
#define MPN_COPY_DECR(dst, src, n)               \
        do                                           \
        {                                            \
            ASSERT((n) >= 0);                        \
            ASSERT(MPN_SAME_OR_DECR_P(dst, src, n)); \
            if ((n) != 0)                            \
            {                                        \
                mp_size_t __n = (n) - 1;             \
                mp_ptr __dst = (dst) + __n;          \
                mp_srcptr __src = (src) + __n;       \
                mp_limb_t __x;                       \
                __x = *__src--;                      \
                if (__n != 0)                        \
                {                                    \
                    do                               \
                    {                                \
                        *__dst-- = __x;              \
                        __x = *__src--;              \
                    } while (--__n);                 \
                }                                    \
                *__dst-- = __x;                      \
            }                                        \
        } while (0)
#endif

#ifndef MPN_COPY
#define MPN_COPY(d, s, n)                        \
        do                                           \
        {                                            \
            ASSERT(MPN_SAME_OR_SEPARATE_P(d, s, n)); \
            MPN_COPY_INCR(d, s, n);                  \
        } while (0)
#endif

        /* Set {dst,size} to the limbs of {src,size} in reverse order. */
#define MPN_REVERSE(dst, src, size)                   \
        do                                                \
        {                                                 \
            mp_ptr __dst = (dst);                         \
            mp_size_t __size = (size);                    \
            mp_srcptr __src = (src) + __size - 1;         \
            mp_size_t __i;                                \
            ASSERT((size) >= 0);                          \
            ASSERT(!MPN_OVERLAP_P(dst, size, src, size)); \
            CRAY_Pragma("_CRI ivdep");                    \
            for (__i = 0; __i < __size; __i++)            \
            {                                             \
                *__dst = *__src;                          \
                __dst++;                                  \
                __src--;                                  \
            }                                             \
        } while (0)

        /* Zero n limbs at dst.

           For power and powerpc we want an inline stu/bdnz loop for zeroing.  On
           ppc630 for instance this is optimal since it can sustain only 1 store per
           cycle.

           gcc 2.95.x (for powerpc64 -maix64, or powerpc32) doesn't recognise the
           "for" loop in the generic code below can become stu/bdnz.  The do/while
           here helps it get to that.  The same caveat about plain -mpowerpc64 mode
           applies here as to __GMPN_COPY_INCR in gmp.h.

           xlc 3.1 already generates stu/bdnz from the generic C, and does so from
           this loop too.

           Enhancement: GLIBC does some trickery with dcbz to zero whole cache lines
           at a time.  MPN_ZERO isn't all that important in GMP, so it might be more
           trouble than it's worth to do the same, though perhaps a call to memset
           would be good when on a GNU system.  */

#if HAVE_HOST_CPU_FAMILY_power || HAVE_HOST_CPU_FAMILY_powerpc
#define MPN_FILL(dst, n, f)       \
        do                            \
        {                             \
            mp_ptr __dst = (dst) - 1; \
            mp_size_t __n = (n);      \
            ASSERT(__n > 0);          \
            do                        \
                *++__dst = (f);       \
            while (--__n);            \
        } while (0)
#endif

#ifndef MPN_FILL
#define MPN_FILL(dst, n, f)   \
        do                        \
        {                         \
            mp_ptr __dst = (dst); \
            mp_size_t __n = (n);  \
            ASSERT(__n > 0);      \
            do                    \
                *__dst++ = (f);   \
            while (--__n);        \
        } while (0)
#endif

#define MPN_ZERO(dst, n)                    \
        do                                      \
        {                                       \
            ASSERT((n) >= 0);                   \
            if ((n) != 0)                       \
                MPN_FILL(dst, n, CNST_LIMB(0)); \
        } while (0)

        /* On the x86s repe/scasl doesn't seem useful, since it takes many cycles to
           start up and would need to strip a lot of zeros before it'd be faster
           than a simple cmpl loop.  Here are some times in cycles for
           std/repe/scasl/cld and cld/repe/scasl (the latter would be for stripping
           low zeros).

                std   cld
               P5    18    16
               P6    46    38
               K6    36    13
               K7    21    20
        */
#ifndef MPN_NORMALIZE
#define MPN_NORMALIZE(DST, NLIMBS)        \
        do                                    \
        {                                     \
            while ((NLIMBS) > 0)              \
            {                                 \
                if ((DST)[(NLIMBS) - 1] != 0) \
                    break;                    \
                (NLIMBS)--;                   \
            }                                 \
        } while (0)
#endif
#ifndef MPN_NORMALIZE_NOT_ZERO
#define MPN_NORMALIZE_NOT_ZERO(DST, NLIMBS) \
        do                                      \
        {                                       \
            while (1)                           \
            {                                   \
                ASSERT((NLIMBS) >= 1);          \
                if ((DST)[(NLIMBS) - 1] != 0)   \
                    break;                      \
                (NLIMBS)--;                     \
            }                                   \
        } while (0)
#endif

        /* Strip least significant zero limbs from {ptr,size} by incrementing ptr
           and decrementing size.  low should be ptr[0], and will be the new ptr[0]
           on returning.  The number in {ptr,size} must be non-zero, ie. size!=0 and
           somewhere a non-zero limb.  */
#define MPN_STRIP_LOW_ZEROS_NOT_ZERO(ptr, size, low) \
        do                                               \
        {                                                \
            ASSERT((size) >= 1);                         \
            ASSERT((low) == (ptr)[0]);                   \
                                                         \
            while ((low) == 0)                           \
            {                                            \
                (size)--;                                \
                ASSERT((size) >= 1);                     \
                (ptr)++;                                 \
                (low) = *(ptr);                          \
            }                                            \
        } while (0)

        /* Initialize X of type mpz_t with space for NLIMBS limbs.  X should be a
           temporary variable; it will be automatically cleared out at function
           return.  We use __x here to make it possible to accept both mpz_ptr and
           mpz_t arguments.  */
#define MPZ_TMP_INIT(X, NLIMBS)               \
        do                                        \
        {                                         \
            mpz_ptr __x = (X);                    \
            ASSERT((NLIMBS) >= 1);                \
            __x->_mp_alloc = (NLIMBS);            \
            __x->_mp_d = TMP_ALLOC_LIMBS(NLIMBS); \
        } while (0)

#if WANT_ASSERT
        static inline void *
        _mpz_newalloc(mpz_ptr z, mp_size_t n)
        {
            void *res = _mpz_realloc(z, n);
            /* If we are checking the code, force a random change to limbs. */
            ((mp_ptr)res)[0] = ~((mp_ptr)res)[ALLOC(z) - 1];
            return res;
        }
#else
#define _mpz_newalloc _mpz_realloc
#endif
        /* Realloc for an mpz_t WHAT if it has less than NEEDED limbs.  */
#define MPZ_REALLOC(z, n) (UNLIKELY((n) > ALLOC(z))         \
                                   ? (mp_ptr)_mpz_realloc(z, n) \
                                   : PTR(z))
#define MPZ_NEWALLOC(z, n) (UNLIKELY((n) > ALLOC(z))          \
                                    ? (mp_ptr)_mpz_newalloc(z, n) \
                                    : PTR(z))

#define MPZ_EQUAL_1_P(z) (SIZ(z) == 1 && PTR(z)[0] == 1)

        /* MPN_FIB2_SIZE(n) is the size in limbs required by gpmpn_fib2_ui for fp and
           f1p.

           From Knuth vol 1 section 1.2.8, F[n] = phi^n/sqrt(5) rounded to the
           nearest integer, where phi=(1+sqrt(5))/2 is the golden ratio.  So the
           number of bits required is n*log_2((1+sqrt(5))/2) = n*0.6942419.

           The multiplier used is 23/32=0.71875 for efficient calculation on CPUs
           without good floating point.  There's +2 for rounding up, and a further
           +2 since at the last step x limbs are doubled into a 2x+1 limb region
           whereas the actual F[2k] value might be only 2x-1 limbs.

           Note that a division is done first, since on a 32-bit system it's at
           least conceivable to go right up to n==ULONG_MAX.  (F[2^32-1] would be
           about 380Mbytes, plus temporary workspace of about 1.2Gbytes here and
           whatever a multiply of two 190Mbyte numbers takes.)

           Enhancement: When GMP_NUMB_BITS is not a power of 2 the division could be
           worked into the multiplier.  */

#define MPN_FIB2_SIZE(n) \
        ((mp_size_t)((n) / 32 * 23 / GMP_NUMB_BITS) + 4)

        /* FIB_TABLE(n) returns the Fibonacci number F[n].  Must have n in the range
           -1 <= n <= FIB_TABLE_LIMIT (that constant in fib_table.h).

           FIB_TABLE_LUCNUM_LIMIT (in fib_table.h) is the largest n for which L[n] =
           F[n] + 2*F[n-1] fits in a limb.  */

#define FIB_TABLE(n) (__gmp_fib_table[(n) + 1])

        extern const mp_limb_t __gmp_oddfac_table[];
        extern const mp_limb_t __gmp_odd2fac_table[];
        extern const unsigned char __gmp_fac2cnt_table[];
        extern const mp_limb_t __gmp_limbroots_table[];

        /* n^log <= GMP_NUMB_MAX, a limb can store log factors less than n */
        static inline unsigned
        log_n_max(mp_limb_t n)
        {
            unsigned log;
            for (log = 8; n > __gmp_limbroots_table[log - 1]; log--)
                ;
            return log;
        }

#define SIEVESIZE 512 /* FIXME: Allow gpgmp_init_primesieve to choose */
        typedef struct
        {
            unsigned long d;                /* current index in s[] */
            unsigned long s0;               /* number corresponding to s[0] */
            unsigned long sqrt_s0;          /* misnomer for sqrt(s[SIEVESIZE-1]) */
            unsigned char s[SIEVESIZE + 1]; /* sieve table */
        } gmp_primesieve_t;




#define gpgmp_init_primesieve __gpgmp_init_primesieve
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpgmp_init_primesieve(gmp_primesieve_t *);

#define gpgmp_nextprime __gpgmp_nextprime
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE unsigned long int gpgmp_nextprime(gmp_primesieve_t *);

#define gmp_primesieve __gmp_primesieve
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gmp_primesieve(mp_ptr, mp_limb_t);




#ifndef MUL_TOOM22_THRESHOLD
#define MUL_TOOM22_THRESHOLD 30
#endif

#ifndef MUL_TOOM33_THRESHOLD
#define MUL_TOOM33_THRESHOLD 100
#endif

#ifndef MUL_TOOM44_THRESHOLD
#define MUL_TOOM44_THRESHOLD 300
#endif

#ifndef MUL_TOOM6H_THRESHOLD
#define MUL_TOOM6H_THRESHOLD 350
#endif

#ifndef SQR_TOOM6_THRESHOLD
#define SQR_TOOM6_THRESHOLD MUL_TOOM6H_THRESHOLD
#endif

#ifndef MUL_TOOM8H_THRESHOLD
#define MUL_TOOM8H_THRESHOLD 450
#endif

#ifndef SQR_TOOM8_THRESHOLD
#define SQR_TOOM8_THRESHOLD MUL_TOOM8H_THRESHOLD
#endif

#ifndef MUL_TOOM32_TO_TOOM43_THRESHOLD
#define MUL_TOOM32_TO_TOOM43_THRESHOLD 100
#endif

#ifndef MUL_TOOM32_TO_TOOM53_THRESHOLD
#define MUL_TOOM32_TO_TOOM53_THRESHOLD 110
#endif

#ifndef MUL_TOOM42_TO_TOOM53_THRESHOLD
#define MUL_TOOM42_TO_TOOM53_THRESHOLD 100
#endif

#ifndef MUL_TOOM42_TO_TOOM63_THRESHOLD
#define MUL_TOOM42_TO_TOOM63_THRESHOLD 110
#endif

#ifndef MUL_TOOM43_TO_TOOM54_THRESHOLD
#define MUL_TOOM43_TO_TOOM54_THRESHOLD 150
#endif

        /* MUL_TOOM22_THRESHOLD_LIMIT is the maximum for MUL_TOOM22_THRESHOLD.  In a
           normal build MUL_TOOM22_THRESHOLD is a constant and we use that.  In a fat
           binary or tune program build MUL_TOOM22_THRESHOLD is a variable and a
           separate hard limit will have been defined.  Similarly for TOOM3.  */
#ifndef MUL_TOOM22_THRESHOLD_LIMIT
#define MUL_TOOM22_THRESHOLD_LIMIT MUL_TOOM22_THRESHOLD
#endif
#ifndef MUL_TOOM33_THRESHOLD_LIMIT
#define MUL_TOOM33_THRESHOLD_LIMIT MUL_TOOM33_THRESHOLD
#endif
#ifndef MULLO_BASECASE_THRESHOLD_LIMIT
#define MULLO_BASECASE_THRESHOLD_LIMIT MULLO_BASECASE_THRESHOLD
#endif
#ifndef SQRLO_BASECASE_THRESHOLD_LIMIT
#define SQRLO_BASECASE_THRESHOLD_LIMIT SQRLO_BASECASE_THRESHOLD
#endif
#ifndef SQRLO_DC_THRESHOLD_LIMIT
#define SQRLO_DC_THRESHOLD_LIMIT SQRLO_DC_THRESHOLD
#endif

        /* SQR_BASECASE_THRESHOLD is where gpmpn_sqr_basecase should take over from
           gpmpn_mul_basecase.  Default is to use gpmpn_sqr_basecase from 0.  (Note that we
           certainly always want it if there's a native assembler gpmpn_sqr_basecase.)

           If it turns out that gpmpn_toom2_sqr becomes faster than gpmpn_mul_basecase
           before gpmpn_sqr_basecase does, then SQR_BASECASE_THRESHOLD is the toom2
           threshold and SQR_TOOM2_THRESHOLD is 0.  This oddity arises more or less
           because SQR_TOOM2_THRESHOLD represents the size up to which gpmpn_sqr_basecase
           should be used, and that may be never.  */

#ifndef SQR_BASECASE_THRESHOLD
#define SQR_BASECASE_THRESHOLD 0 /* never use gpmpn_mul_basecase */
#endif

#ifndef SQR_TOOM2_THRESHOLD
#define SQR_TOOM2_THRESHOLD 50
#endif

#ifndef SQR_TOOM3_THRESHOLD
#define SQR_TOOM3_THRESHOLD 120
#endif

#ifndef SQR_TOOM4_THRESHOLD
#define SQR_TOOM4_THRESHOLD 400
#endif

        /* See comments above about MUL_TOOM33_THRESHOLD_LIMIT.  */
#ifndef SQR_TOOM3_THRESHOLD_LIMIT
#define SQR_TOOM3_THRESHOLD_LIMIT SQR_TOOM3_THRESHOLD
#endif

#ifndef MULMID_TOOM42_THRESHOLD
#define MULMID_TOOM42_THRESHOLD MUL_TOOM22_THRESHOLD
#endif

#ifndef MULLO_BASECASE_THRESHOLD
#define MULLO_BASECASE_THRESHOLD 0 /* never use gpmpn_mul_basecase */
#endif

#ifndef MULLO_DC_THRESHOLD
#define MULLO_DC_THRESHOLD (2 * MUL_TOOM22_THRESHOLD)
#endif

#ifndef MULLO_MUL_N_THRESHOLD
#define MULLO_MUL_N_THRESHOLD (2 * MUL_FFT_THRESHOLD)
#endif

#ifndef SQRLO_BASECASE_THRESHOLD
#define SQRLO_BASECASE_THRESHOLD 0 /* never use gpmpn_sqr_basecase */
#endif

#ifndef SQRLO_DC_THRESHOLD
#define SQRLO_DC_THRESHOLD (MULLO_DC_THRESHOLD)
#endif

#ifndef SQRLO_SQR_THRESHOLD
#define SQRLO_SQR_THRESHOLD (MULLO_MUL_N_THRESHOLD)
#endif

#ifndef DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD (2 * MUL_TOOM22_THRESHOLD)
#endif

#ifndef DC_DIVAPPR_Q_THRESHOLD
#define DC_DIVAPPR_Q_THRESHOLD 200
#endif

#ifndef DC_BDIV_QR_THRESHOLD
#define DC_BDIV_QR_THRESHOLD (2 * MUL_TOOM22_THRESHOLD)
#endif

#ifndef DC_BDIV_Q_THRESHOLD
#define DC_BDIV_Q_THRESHOLD 180
#endif

#ifndef DIVEXACT_JEB_THRESHOLD
#define DIVEXACT_JEB_THRESHOLD 25
#endif

#ifndef INV_MULMOD_BNM1_THRESHOLD
#define INV_MULMOD_BNM1_THRESHOLD (4 * MULMOD_BNM1_THRESHOLD)
#endif

#ifndef INV_APPR_THRESHOLD
#define INV_APPR_THRESHOLD INV_NEWTON_THRESHOLD
#endif

#ifndef INV_NEWTON_THRESHOLD
#define INV_NEWTON_THRESHOLD 200
#endif

#ifndef BINV_NEWTON_THRESHOLD
#define BINV_NEWTON_THRESHOLD 300
#endif

#ifndef MU_DIVAPPR_Q_THRESHOLD
#define MU_DIVAPPR_Q_THRESHOLD 2000
#endif

#ifndef MU_DIV_QR_THRESHOLD
#define MU_DIV_QR_THRESHOLD 2000
#endif

#ifndef MUPI_DIV_QR_THRESHOLD
#define MUPI_DIV_QR_THRESHOLD 200
#endif

#ifndef MU_BDIV_Q_THRESHOLD
#define MU_BDIV_Q_THRESHOLD 2000
#endif

#ifndef MU_BDIV_QR_THRESHOLD
#define MU_BDIV_QR_THRESHOLD 2000
#endif

#ifndef MULMOD_BNM1_THRESHOLD
#define MULMOD_BNM1_THRESHOLD 16
#endif

#ifndef SQRMOD_BNM1_THRESHOLD
#define SQRMOD_BNM1_THRESHOLD 16
#endif

#ifndef MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD
#define MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD (INV_MULMOD_BNM1_THRESHOLD / 2)
#endif

#if HAVE_NATIVE_gpmpn_addmul_2 || HAVE_NATIVE_gpmpn_redc_2

#ifndef REDC_1_TO_REDC_2_THRESHOLD
#define REDC_1_TO_REDC_2_THRESHOLD 15
#endif
#ifndef REDC_2_TO_REDC_N_THRESHOLD
#define REDC_2_TO_REDC_N_THRESHOLD 100
#endif

#else

#ifndef REDC_1_TO_REDC_N_THRESHOLD
#define REDC_1_TO_REDC_N_THRESHOLD 100
#endif

#endif /* HAVE_NATIVE_gpmpn_addmul_2 || HAVE_NATIVE_gpmpn_redc_2 */

        /* First k to use for an FFT modF multiply.  A modF FFT is an order
           log(2^k)/log(2^(k-1)) algorithm, so k=3 is merely 1.5 like karatsuba,
           whereas k=4 is 1.33 which is faster than toom3 at 1.485.    */
#define FFT_FIRST_K 4

        /* Threshold at which FFT should be used to do a modF NxN -> N multiply. */
#ifndef MUL_FFT_MODF_THRESHOLD
#define MUL_FFT_MODF_THRESHOLD (MUL_TOOM33_THRESHOLD * 3)
#endif
#ifndef SQR_FFT_MODF_THRESHOLD
#define SQR_FFT_MODF_THRESHOLD (SQR_TOOM3_THRESHOLD * 3)
#endif

        /* Threshold at which FFT should be used to do an NxN -> 2N multiply.  This
           will be a size where FFT is using k=7 or k=8, since an FFT-k used for an
           NxN->2N multiply and not recursing into itself is an order
           log(2^k)/log(2^(k-2)) algorithm, so it'll be at least k=7 at 1.39 which
           is the first better than toom3.  */
#ifndef MUL_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD (MUL_FFT_MODF_THRESHOLD * 10)
#endif
#ifndef SQR_FFT_THRESHOLD
#define SQR_FFT_THRESHOLD (SQR_FFT_MODF_THRESHOLD * 10)
#endif

        /* Table of thresholds for successive modF FFT "k"s.  The first entry is
           where FFT_FIRST_K+1 should be used, the second FFT_FIRST_K+2,
           etc.  See gpmpn_fft_best_k(). */
#ifndef MUL_FFT_TABLE
#define MUL_FFT_TABLE                       \
        {MUL_TOOM33_THRESHOLD * 4,   /* k=5 */  \
         MUL_TOOM33_THRESHOLD * 8,   /* k=6 */  \
         MUL_TOOM33_THRESHOLD * 16,  /* k=7 */  \
         MUL_TOOM33_THRESHOLD * 32,  /* k=8 */  \
         MUL_TOOM33_THRESHOLD * 96,  /* k=9 */  \
         MUL_TOOM33_THRESHOLD * 288, /* k=10 */ \
         0}
#endif
#ifndef SQR_FFT_TABLE
#define SQR_FFT_TABLE                      \
        {SQR_TOOM3_THRESHOLD * 4,   /* k=5 */  \
         SQR_TOOM3_THRESHOLD * 8,   /* k=6 */  \
         SQR_TOOM3_THRESHOLD * 16,  /* k=7 */  \
         SQR_TOOM3_THRESHOLD * 32,  /* k=8 */  \
         SQR_TOOM3_THRESHOLD * 96,  /* k=9 */  \
         SQR_TOOM3_THRESHOLD * 288, /* k=10 */ \
         0}
#endif

        struct fft_table_nk
        {
            gmp_uint_least32_t n : 27;
            gmp_uint_least32_t k : 5;
        };

#ifndef FFT_TABLE_ATTRS
#define FFT_TABLE_ATTRS static const
#endif

#define MPN_FFT_TABLE_SIZE 16

#ifndef DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD (3 * MUL_TOOM22_THRESHOLD)
#endif

#ifndef GET_STR_DC_THRESHOLD
#define GET_STR_DC_THRESHOLD 18
#endif

#ifndef GET_STR_PRECOMPUTE_THRESHOLD
#define GET_STR_PRECOMPUTE_THRESHOLD 35
#endif

#ifndef SET_STR_DC_THRESHOLD
#define SET_STR_DC_THRESHOLD 750
#endif

#ifndef SET_STR_PRECOMPUTE_THRESHOLD
#define SET_STR_PRECOMPUTE_THRESHOLD 2000
#endif

#ifndef FAC_ODD_THRESHOLD
#define FAC_ODD_THRESHOLD 35
#endif

#ifndef FAC_DSC_THRESHOLD
#define FAC_DSC_THRESHOLD 400
#endif

        /* Return non-zero if xp,xsize and yp,ysize overlap.
           If xp+xsize<=yp there's no overlap, or if yp+ysize<=xp there's no
           overlap.  If both these are false, there's an overlap. */
#define MPN_OVERLAP_P(xp, xsize, yp, ysize) \
        ((xp) + (xsize) > (yp) && (yp) + (ysize) > (xp))
#define MEM_OVERLAP_P(xp, xsize, yp, ysize) \
        ((char *)(xp) + (xsize) > (char *)(yp) && (char *)(yp) + (ysize) > (char *)(xp))

        /* Return non-zero if xp,xsize and yp,ysize are either identical or not
           overlapping.  Return zero if they're partially overlapping. */
#define MPN_SAME_OR_SEPARATE_P(xp, yp, size) \
        MPN_SAME_OR_SEPARATE2_P(xp, size, yp, size)
#define MPN_SAME_OR_SEPARATE2_P(xp, xsize, yp, ysize) \
        ((xp) == (yp) || !MPN_OVERLAP_P(xp, xsize, yp, ysize))

        /* Return non-zero if dst,dsize and src,ssize are either identical or
           overlapping in a way suitable for an incrementing/decrementing algorithm.
           Return zero if they're partially overlapping in an unsuitable fashion. */
#define MPN_SAME_OR_INCR2_P(dst, dsize, src, ssize) \
        ((dst) <= (src) || !MPN_OVERLAP_P(dst, dsize, src, ssize))
#define MPN_SAME_OR_INCR_P(dst, src, size) \
        MPN_SAME_OR_INCR2_P(dst, size, src, size)
#define MPN_SAME_OR_DECR2_P(dst, dsize, src, ssize) \
        ((dst) >= (src) || !MPN_OVERLAP_P(dst, dsize, src, ssize))
#define MPN_SAME_OR_DECR_P(dst, src, size) \
        MPN_SAME_OR_DECR2_P(dst, size, src, size)

        /* ASSERT() is a private assertion checking scheme, similar to <assert.h>.
           ASSERT() does the check only if WANT_ASSERT is selected, ASSERT_ALWAYS()
           does it always.  Generally assertions are meant for development, but
           might help when looking for a problem later too.  */

#ifdef __LINE__
#define ASSERT_LINE __LINE__
#else
#define ASSERT_LINE -1
#endif

#ifdef __FILE__
#define ASSERT_FILE __FILE__
#else
#define ASSERT_FILE ""
#endif

__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_assert_header(const char *, int);

__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_assert_fail (const char *, int , const char *);


#define ASSERT_FAIL(expr) __gpgmp_assert_fail(ASSERT_FILE, ASSERT_LINE, #expr)

#define ASSERT_ALWAYS(expr)    \
        do                         \
        {                          \
            if (UNLIKELY(!(expr))) \
                ASSERT_FAIL(expr); \
        } while (0)

#if WANT_ASSERT
#define ASSERT(expr) ASSERT_ALWAYS(expr)
#else
#define ASSERT(expr) \
        do               \
        {                \
        } while (0)
#endif

        /* ASSERT_CARRY checks the expression is non-zero, and ASSERT_NOCARRY checks
           that it's zero.  In both cases if assertion checking is disabled the
           expression is still evaluated.  These macros are meant for use with
           routines like gpmpn_add_n() where the return value represents a carry or
           whatever that should or shouldn't occur in some context.  For example,
           ASSERT_NOCARRY (gpmpn_add_n (rp, s1p, s2p, size)); */
#if WANT_ASSERT
#define ASSERT_CARRY(expr) ASSERT_ALWAYS((expr) != 0)
#define ASSERT_NOCARRY(expr) ASSERT_ALWAYS((expr) == 0)
#else
#define ASSERT_CARRY(expr) (expr)
#define ASSERT_NOCARRY(expr) (expr)
#endif

        /* ASSERT_CODE includes code when assertion checking is wanted.  This is the
           same as writing "#if WANT_ASSERT", but more compact.  */
#if WANT_ASSERT
#define ASSERT_CODE(expr) expr
#else
#define ASSERT_CODE(expr)
#endif

        /* Test that an mpq_t is in fully canonical form.  This can be used as
           protection on routines like mpq_equal which give wrong results on
           non-canonical inputs.  */
#if WANT_ASSERT
#define ASSERT_MPQ_CANONICAL(q)                         \
        do                                                  \
        {                                                   \
            ASSERT(q->_mp_den._mp_size > 0);                \
            if (q->_mp_num._mp_size == 0)                   \
            {                                               \
                /* zero should be 0/1 */                    \
                ASSERT(mpz_cmp_ui(mpq_denref(q), 1L) == 0); \
            }                                               \
            else                                            \
            {                                               \
                /* no common factors */                     \
                mpz_t __g;                                  \
                mpz_init(__g);                              \
                mpz_gcd(__g, mpq_numref(q), mpq_denref(q)); \
                ASSERT(mpz_cmp_ui(__g, 1) == 0);            \
                mpz_clear(__g);                             \
            }                                               \
        } while (0)
#else
#define ASSERT_MPQ_CANONICAL(q) \
        do                          \
        {                           \
        } while (0)
#endif

        /* Check that the nail parts are zero. */
#define ASSERT_ALWAYS_LIMB(limb)                   \
        do                                             \
        {                                              \
            mp_limb_t __nail = (limb) & GMP_NAIL_MASK; \
            ASSERT_ALWAYS(__nail == 0);                \
        } while (0)
#define ASSERT_ALWAYS_MPN(ptr, size)               \
        do                                             \
        {                                              \
            /* let whole loop go dead when no nails */ \
            if (GMP_NAIL_BITS != 0)                    \
            {                                          \
                mp_size_t __i;                         \
                for (__i = 0; __i < (size); __i++)     \
                    ASSERT_ALWAYS_LIMB((ptr)[__i]);    \
            }                                          \
        } while (0)
#if WANT_ASSERT
#define ASSERT_LIMB(limb) ASSERT_ALWAYS_LIMB(limb)
#define ASSERT_MPN(ptr, size) ASSERT_ALWAYS_MPN(ptr, size)
#else
#define ASSERT_LIMB(limb) \
        do                    \
        {                     \
        } while (0)
#define ASSERT_MPN(ptr, size) \
        do                        \
        {                         \
        } while (0)
#endif

        /* Assert that an mpn region {ptr,size} is zero, or non-zero.
           size==0 is allowed, and in that case {ptr,size} considered to be zero.  */
#if WANT_ASSERT
#define ASSERT_MPN_ZERO_P(ptr, size)       \
        do                                     \
        {                                      \
            mp_size_t __i;                     \
            ASSERT((size) >= 0);               \
            for (__i = 0; __i < (size); __i++) \
                ASSERT((ptr)[__i] == 0);       \
        } while (0)
#define ASSERT_MPN_NONZERO_P(ptr, size)    \
        do                                     \
        {                                      \
            mp_size_t __i;                     \
            int __nonzero = 0;                 \
            ASSERT((size) >= 0);               \
            for (__i = 0; __i < (size); __i++) \
                if ((ptr)[__i] != 0)           \
                {                              \
                    __nonzero = 1;             \
                    break;                     \
                }                              \
            ASSERT(__nonzero);                 \
        } while (0)
#else
#define ASSERT_MPN_ZERO_P(ptr, size) \
        do                               \
        {                                \
        } while (0)
#define ASSERT_MPN_NONZERO_P(ptr, size) \
        do                                  \
        {                                   \
        } while (0)
#endif

#if !HAVE_NATIVE_gpmpn_com
#undef gpmpn_com
#define gpmpn_com(d, s, n)                               \
        do                                                 \
        {                                                  \
            mp_ptr __d = (d);                              \
            mp_srcptr __s = (s);                           \
            mp_size_t __n = (n);                           \
            ASSERT(__n >= 1);                              \
            ASSERT(MPN_SAME_OR_SEPARATE_P(__d, __s, __n)); \
            do                                             \
                *__d++ = (~*__s++) & GMP_NUMB_MASK;        \
            while (--__n);                                 \
        } while (0)
#endif

#define MPN_LOGOPS_N_INLINE(rp, up, vp, n, operation)    \
        do                                                   \
        {                                                    \
            mp_srcptr __up = (up);                           \
            mp_srcptr __vp = (vp);                           \
            mp_ptr __rp = (rp);                              \
            mp_size_t __n = (n);                             \
            mp_limb_t __a, __b;                              \
            ASSERT(__n > 0);                                 \
            ASSERT(MPN_SAME_OR_SEPARATE_P(__rp, __up, __n)); \
            ASSERT(MPN_SAME_OR_SEPARATE_P(__rp, __vp, __n)); \
            __up += __n;                                     \
            __vp += __n;                                     \
            __rp += __n;                                     \
            __n = -__n;                                      \
            do                                               \
            {                                                \
                __a = __up[__n];                             \
                __b = __vp[__n];                             \
                __rp[__n] = operation;                       \
            } while (++__n);                                 \
        } while (0)

#if !HAVE_NATIVE_gpmpn_and_n
#undef gpmpn_and_n
#define gpmpn_and_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, __a &__b)
#endif

#if !HAVE_NATIVE_gpmpn_andn_n
#undef gpmpn_andn_n
#define gpmpn_andn_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, __a & ~__b)
#endif

#if !HAVE_NATIVE_gpmpn_nand_n
#undef gpmpn_nand_n
#define gpmpn_nand_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, ~(__a & __b) & GMP_NUMB_MASK)
#endif

#if !HAVE_NATIVE_gpmpn_ior_n
#undef gpmpn_ior_n
#define gpmpn_ior_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, __a | __b)
#endif

#if !HAVE_NATIVE_gpmpn_iorn_n
#undef gpmpn_iorn_n
#define gpmpn_iorn_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, (__a | ~__b) & GMP_NUMB_MASK)
#endif

#if !HAVE_NATIVE_gpmpn_nior_n
#undef gpmpn_nior_n
#define gpmpn_nior_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, ~(__a | __b) & GMP_NUMB_MASK)
#endif

#if !HAVE_NATIVE_gpmpn_xor_n
#undef gpmpn_xor_n
#define gpmpn_xor_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, __a ^ __b)
#endif

#if !HAVE_NATIVE_gpmpn_xnor_n
#undef gpmpn_xnor_n
#define gpmpn_xnor_n(rp, up, vp, n) \
        MPN_LOGOPS_N_INLINE(rp, up, vp, n, ~(__a ^ __b) & GMP_NUMB_MASK)
#endif
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_trialdiv __GPGMP_MPN(trialdiv)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_trialdiv(mp_srcptr, mp_size_t, mp_size_t, int *);

#define gpmpn_remove __GPGMP_MPN(remove)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_remove(mp_ptr, mp_size_t *, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_bitcnt_t);
GPGMP_MPN_NAMESPACE_END
        /* ADDC_LIMB sets w=x+y and cout to 0 or 1 for a carry from that addition. */
#if GMP_NAIL_BITS == 0
#define ADDC_LIMB(cout, w, x, y)   \
        do                             \
        {                              \
            mp_limb_t __x = (x);       \
            mp_limb_t __y = (y);       \
            mp_limb_t __w = __x + __y; \
            (w) = __w;                 \
            (cout) = __w < __x;        \
        } while (0)
#else
#define ADDC_LIMB(cout, w, x, y)       \
        do                                 \
        {                                  \
            mp_limb_t __w;                 \
            ASSERT_LIMB(x);                \
            ASSERT_LIMB(y);                \
            __w = (x) + (y);               \
            (w) = __w & GMP_NUMB_MASK;     \
            (cout) = __w >> GMP_NUMB_BITS; \
        } while (0)
#endif

        /* SUBC_LIMB sets w=x-y and cout to 0 or 1 for a borrow from that
           subtract.  */
#if GMP_NAIL_BITS == 0
#define SUBC_LIMB(cout, w, x, y)   \
        do                             \
        {                              \
            mp_limb_t __x = (x);       \
            mp_limb_t __y = (y);       \
            mp_limb_t __w = __x - __y; \
            (w) = __w;                 \
            (cout) = __w > __x;        \
        } while (0)
#else
#define SUBC_LIMB(cout, w, x, y)             \
        do                                       \
        {                                        \
            mp_limb_t __w = (x) - (y);           \
            (w) = __w & GMP_NUMB_MASK;           \
            (cout) = __w >> (GMP_LIMB_BITS - 1); \
        } while (0)
#endif

        /* MPN_INCR_U does {ptr,size} += n, MPN_DECR_U does {ptr,size} -= n, both
           expecting no carry (or borrow) from that.

           The size parameter is only for the benefit of assertion checking.  In a
           normal build it's unused and the carry/borrow is just propagated as far
           as it needs to go.

           On random data, usually only one or two limbs of {ptr,size} get updated,
           so there's no need for any sophisticated looping, just something compact
           and sensible.

           FIXME: Switch all code from gpmpn_{incr,decr}_u to MPN_{INCR,DECR}_U,
           declaring their operand sizes, then remove the former.  This is purely
           for the benefit of assertion checking.  */

#if defined(__GNUC__) && GMP_NAIL_BITS == 0 && !defined(NO_ASM) && (defined(HAVE_HOST_CPU_FAMILY_x86) || defined(HAVE_HOST_CPU_FAMILY_x86_64)) && !WANT_ASSERT
        /* Better flags handling than the generic C gives on i386, saving a few
           bytes of code and maybe a cycle or two.  */

#define MPN_IORD_U(ptr, incr, aors)                                                                                    \
        do                                                                                                                 \
        {                                                                                                                  \
            mp_ptr __ptr_dummy;                                                                                            \
            if (__builtin_constant_p(incr) && (incr) == 0)                                                                 \
            {                                                                                                              \
            }                                                                                                              \
            else if (__builtin_constant_p(incr) && (incr) == 1)                                                            \
            {                                                                                                              \
                __asm__ __volatile__("\n" ASM_L(top) ":\n"                                                                 \
                                                     "\t" aors "\t$1, (%0)\n"                                              \
                                                     "\tlea\t%c2(%0), %0\n"                                                \
                                                     "\tjc\t" ASM_L(top)                                                   \
                                     : "=r"(__ptr_dummy)                                                                   \
                                     : "0"(ptr), "n"(sizeof(mp_limb_t))                                                    \
                                     : "memory");                                                                          \
            }                                                                                                              \
            else                                                                                                           \
            {                                                                                                              \
                __asm__ __volatile__(aors "\t%2, (%0)\n"                                                                   \
                                          "\tjnc\t" ASM_L(done) "\n" ASM_L(top) ":\n"                                      \
                                                                                "\t" aors "\t$1, %c3(%0)\n"                \
                                                                                "\tlea\t%c3(%0), %0\n"                     \
                                                                                "\tjc\t" ASM_L(top) "\n" ASM_L(done) ":\n" \
                                     : "=r"(__ptr_dummy)                                                                   \
                                     : "0"(ptr),                                                                           \
                                       "re"((mp_limb_t)(incr)), "n"(sizeof(mp_limb_t))                                     \
                                     : "memory");                                                                          \
            }                                                                                                              \
        } while (0)

#if GMP_LIMB_BITS == 32
#define MPN_INCR_U(ptr, size, incr) MPN_IORD_U(ptr, incr, "addl")
#define MPN_DECR_U(ptr, size, incr) MPN_IORD_U(ptr, incr, "subl")
#endif
#if GMP_LIMB_BITS == 64
#define MPN_INCR_U(ptr, size, incr) MPN_IORD_U(ptr, incr, "addq")
#define MPN_DECR_U(ptr, size, incr) MPN_IORD_U(ptr, incr, "subq")
#endif
#define gpmpn_incr_u(ptr, incr) MPN_INCR_U(ptr, 0, incr)
#define gpmpn_decr_u(ptr, incr) MPN_DECR_U(ptr, 0, incr)
#endif

#if GMP_NAIL_BITS == 0
#ifndef gpmpn_incr_u
#define gpmpn_incr_u(p, incr)                            \
        do                                                 \
        {                                                  \
            mp_limb_t __x;                                 \
            mp_ptr __p = (p);                              \
            if (__builtin_constant_p(incr) && (incr) == 1) \
            {                                              \
                while (++(*(__p++)) == 0)                  \
                    ;                                      \
            }                                              \
            else                                           \
            {                                              \
                __x = *__p + (incr);                       \
                *__p = __x;                                \
                if (__x < (incr))                          \
                    while (++(*(++__p)) == 0)              \
                        ;                                  \
            }                                              \
        } while (0)
#endif
#ifndef gpmpn_decr_u
#define gpmpn_decr_u(p, incr)                            \
        do                                                 \
        {                                                  \
            mp_limb_t __x;                                 \
            mp_ptr __p = (p);                              \
            if (__builtin_constant_p(incr) && (incr) == 1) \
            {                                              \
                while ((*(__p++))-- == 0)                  \
                    ;                                      \
            }                                              \
            else                                           \
            {                                              \
                __x = *__p;                                \
                *__p = __x - (incr);                       \
                if (__x < (incr))                          \
                    while ((*(++__p))-- == 0)              \
                        ;                                  \
            }                                              \
        } while (0)
#endif
#endif

#if GMP_NAIL_BITS >= 1
#ifndef gpmpn_incr_u
#define gpmpn_incr_u(p, incr)                            \
        do                                                 \
        {                                                  \
            mp_limb_t __x;                                 \
            mp_ptr __p = (p);                              \
            if (__builtin_constant_p(incr) && (incr) == 1) \
            {                                              \
                do                                         \
                {                                          \
                    __x = (*__p + 1) & GMP_NUMB_MASK;      \
                    *__p++ = __x;                          \
                } while (__x == 0);                        \
            }                                              \
            else                                           \
            {                                              \
                __x = (*__p + (incr));                     \
                *__p++ = __x & GMP_NUMB_MASK;              \
                if (__x >> GMP_NUMB_BITS != 0)             \
                {                                          \
                    do                                     \
                    {                                      \
                        __x = (*__p + 1) & GMP_NUMB_MASK;  \
                        *__p++ = __x;                      \
                    } while (__x == 0);                    \
                }                                          \
            }                                              \
        } while (0)
#endif
#ifndef gpmpn_decr_u
#define gpmpn_decr_u(p, incr)                             \
        do                                                  \
        {                                                   \
            mp_limb_t __x;                                  \
            mp_ptr __p = (p);                               \
            if (__builtin_constant_p(incr) && (incr) == 1)  \
            {                                               \
                do                                          \
                {                                           \
                    __x = *__p;                             \
                    *__p++ = (__x - 1) & GMP_NUMB_MASK;     \
                } while (__x == 0);                         \
            }                                               \
            else                                            \
            {                                               \
                __x = *__p - (incr);                        \
                *__p++ = __x & GMP_NUMB_MASK;               \
                if (__x >> GMP_NUMB_BITS != 0)              \
                {                                           \
                    do                                      \
                    {                                       \
                        __x = *__p;                         \
                        *__p++ = (__x - 1) & GMP_NUMB_MASK; \
                    } while (__x == 0);                     \
                }                                           \
            }                                               \
        } while (0)
#endif
#endif

#ifndef MPN_INCR_U
#if WANT_ASSERT
#define MPN_INCR_U(ptr, size, n)                      \
        do                                                \
        {                                                 \
            ASSERT((size) >= 1);                          \
            ASSERT_NOCARRY(gpmpn_add_1(ptr, ptr, size, n)); \
        } while (0)
#else
#define MPN_INCR_U(ptr, size, n) gpmpn_incr_u(ptr, n)
#endif
#endif

#ifndef MPN_DECR_U
#if WANT_ASSERT
#define MPN_DECR_U(ptr, size, n)                      \
        do                                                \
        {                                                 \
            ASSERT((size) >= 1);                          \
            ASSERT_NOCARRY(gpmpn_sub_1(ptr, ptr, size, n)); \
        } while (0)
#else
#define MPN_DECR_U(ptr, size, n) gpmpn_decr_u(ptr, n)
#endif
#endif

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

#define mp_bases __GPGMP_MPN(bases)

        /* Compute the number of digits in base for nbits bits, making sure the result
           is never too small.  The two variants of the macro implement the same
           function; the GT2 variant below works just for bases > 2.  */
#define DIGITS_IN_BASE_FROM_BITS(res, nbits, b)            \
        do                                                     \
        {                                                      \
            mp_limb_t _ph, _dummy;                             \
            size_t _nbits = (nbits);                           \
            umul_ppmm(_ph, _dummy, mp_bases[b].logb2, _nbits); \
            _ph += (_dummy + _nbits < _dummy);                 \
            res = _ph + 1;                                     \
        } while (0)
#define DIGITS_IN_BASEGT2_FROM_BITS(res, nbits, b)             \
        do                                                         \
        {                                                          \
            mp_limb_t _ph, _dummy;                                 \
            size_t _nbits = (nbits);                               \
            umul_ppmm(_ph, _dummy, mp_bases[b].logb2 + 1, _nbits); \
            res = _ph + 1;                                         \
        } while (0)

        /* For power of 2 bases this is exact.  For other bases the result is either
           exact or one too big.

           To be exact always it'd be necessary to examine all the limbs of the
           operand, since numbers like 100..000 and 99...999 generally differ only
           in the lowest limb.  It'd be possible to examine just a couple of high
           limbs to increase the probability of being exact, but that doesn't seem
           worth bothering with.  */

#define MPN_SIZEINBASE(result, ptr, size, base)                                   \
        do                                                                            \
        {                                                                             \
            int __lb_base, __cnt;                                                     \
            size_t __totbits;                                                         \
                                                                                      \
            ASSERT((size) >= 0);                                                      \
            ASSERT((base) >= 2);                                                      \
            ASSERT((base) < numberof(mp_bases));                                      \
                                                                                      \
            /* Special case for X == 0.  */                                           \
            if ((size) == 0)                                                          \
                (result) = 1;                                                         \
            else                                                                      \
            {                                                                         \
                /* Calculate the total number of significant bits of X.  */           \
                count_leading_zeros(__cnt, (ptr)[(size) - 1]);                        \
                __totbits = (size_t)(size) * GMP_NUMB_BITS - (__cnt - GMP_NAIL_BITS); \
                                                                                      \
                if (POW2_P(base))                                                     \
                {                                                                     \
                    __lb_base = mp_bases[base].big_base;                              \
                    (result) = (__totbits + __lb_base - 1) / __lb_base;               \
                }                                                                     \
                else                                                                  \
                {                                                                     \
                    DIGITS_IN_BASEGT2_FROM_BITS(result, __totbits, base);             \
                }                                                                     \
            }                                                                         \
        } while (0)

#define MPN_SIZEINBASE_2EXP(result, ptr, size, base2exp)                           \
        do                                                                             \
        {                                                                              \
            int __cnt;                                                                 \
            mp_bitcnt_t __totbits;                                                     \
            ASSERT((size) > 0);                                                        \
            ASSERT((ptr)[(size) - 1] != 0);                                            \
            count_leading_zeros(__cnt, (ptr)[(size) - 1]);                             \
            __totbits = (mp_bitcnt_t)(size) * GMP_NUMB_BITS - (__cnt - GMP_NAIL_BITS); \
            (result) = (__totbits + (base2exp) - 1) / (base2exp);                      \
        } while (0)

        /* bit count to limb count, rounding up */
#define BITS_TO_LIMBS(n) (((n) + (GMP_NUMB_BITS - 1)) / GMP_NUMB_BITS)

        /* MPN_SET_UI sets an mpn (ptr, cnt) to given ui.  MPZ_FAKE_UI creates fake
           mpz_t from ui.  The zp argument must have room for LIMBS_PER_ULONG limbs
           in both cases (LIMBS_PER_ULONG is also defined here.) */
#if BITS_PER_ULONG <= GMP_NUMB_BITS /* need one limb per ulong */

#define LIMBS_PER_ULONG 1
#define MPN_SET_UI(zp, zn, u) \
        (zp)[0] = (u);            \
        (zn) = ((zp)[0] != 0);
#define MPZ_FAKE_UI(z, zp, u) \
        (zp)[0] = (u);            \
        PTR(z) = (zp);            \
        SIZ(z) = ((zp)[0] != 0);  \
        ASSERT_CODE(ALLOC(z) = 1);

#else /* need two limbs per ulong */

#define LIMBS_PER_ULONG 2
#define MPN_SET_UI(zp, zn, u)                   \
        (zp)[0] = (u) & GMP_NUMB_MASK;              \
        (zp)[1] = (u) >> GMP_NUMB_BITS;             \
        (zn) = ((zp)[1] != 0 ? 2 : (zp)[0] != 0 ? 1 \
                                                : 0);
#define MPZ_FAKE_UI(z, zp, u)                       \
        (zp)[0] = (u) & GMP_NUMB_MASK;                  \
        (zp)[1] = (u) >> GMP_NUMB_BITS;                 \
        SIZ(z) = ((zp)[1] != 0 ? 2 : (zp)[0] != 0 ? 1   \
                                                  : 0); \
        PTR(z) = (zp);                                  \
        ASSERT_CODE(ALLOC(z) = 2);

#endif

#if HAVE_HOST_CPU_FAMILY_x86
#define TARGET_REGISTER_STARVED 1
#else
#define TARGET_REGISTER_STARVED 0
#endif

        /* LIMB_HIGHBIT_TO_MASK(n) examines the high bit of a limb value and turns 1
           or 0 there into a limb 0xFF..FF or 0 respectively.

           On most CPUs this is just an arithmetic right shift by GMP_LIMB_BITS-1,
           but C99 doesn't guarantee signed right shifts are arithmetic, so we have
           a little compile-time test and a fallback to a "? :" form.  The latter is
           necessary for instance on Cray vector systems.

           Recent versions of gcc (eg. 3.3) will in fact optimize a "? :" like this
           to an arithmetic right shift anyway, but it's good to get the desired
           shift on past versions too (in particular since an important use of
           LIMB_HIGHBIT_TO_MASK is in udiv_qrnnd_preinv).  */

#define LIMB_HIGHBIT_TO_MASK(n)                         \
        (((mp_limb_signed_t) - 1 >> 1) < 0                  \
             ? (mp_limb_signed_t)(n) >> (GMP_LIMB_BITS - 1) \
         : (n) & GMP_LIMB_HIGHBIT ? MP_LIMB_T_MAX           \
                                  : CNST_LIMB(0))

GPGMP_MPN_NAMESPACE_BEGIN
        /* Use a library function for invert_limb, if available. */
#define gpmpn_invert_limb __GPGMP_MPN(invert_limb)
        //__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_invert_limb(mp_limb_t) ATTRIBUTE_CONST;
GPGMP_MPN_NAMESPACE_END
#if !defined(invert_limb) && HAVE_NATIVE_gpmpn_invert_limb
#define invert_limb(invxl, xl)         \
        do                                 \
        {                                  \
            (invxl) = gpmpn_invert_limb(xl); \
        } while (0)
#endif

#ifndef invert_limb
#define invert_limb(invxl, xl)                               \
        do                                                       \
        {                                                        \
            mp_limb_t _dummy;                                    \
            ASSERT((xl) != 0);                                   \
            udiv_qrnnd(invxl, _dummy, ~(xl), ~CNST_LIMB(0), xl); \
        } while (0)
#endif

#define invert_pi1(dinv, d1, d0)              \
        do                                        \
        {                                         \
            mp_limb_t _v, _p, _t1, _t0, _mask;    \
            invert_limb(_v, d1);                  \
            _p = (d1) * _v;                       \
            _p += (d0);                           \
            if (_p < (d0))                        \
            {                                     \
                _v--;                             \
                _mask = -(mp_limb_t)(_p >= (d1)); \
                _p -= (d1);                       \
                _v += _mask;                      \
                _p -= _mask & (d1);               \
            }                                     \
            umul_ppmm(_t1, _t0, d0, _v);          \
            _p += _t1;                            \
            if (_p < _t1)                         \
            {                                     \
                _v--;                             \
                if (UNLIKELY(_p >= (d1)))         \
                {                                 \
                    if (_p > (d1) || _t0 >= (d0)) \
                        _v--;                     \
                }                                 \
            }                                     \
            (dinv).inv32 = _v;                    \
        } while (0)

        /* udiv_qrnnd_preinv -- Based on work by Niels Möller and Torbjörn Granlund.
           We write things strangely below, to help gcc.  A more straightforward
           version:
            _r = (nl) - _qh * (d);
            _t = _r + (d);
            if (_r >= _ql)
              {
                _qh--;
                _r = _t;
              }
           For one operation shorter critical path, one may want to use this form:
            _p = _qh * (d)
            _s = (nl) + (d);
            _r = (nl) - _p;
            _t = _s - _p;
            if (_r >= _ql)
              {
                _qh--;
                _r = _t;
              }
        */
#define udiv_qrnnd_preinv(q, r, nh, nl, d, di)                         \
        do                                                                 \
        {                                                                  \
            mp_limb_t _qh, _ql, _r, _mask;                                 \
            umul_ppmm(_qh, _ql, (nh), (di));                               \
            if (__builtin_constant_p(nl) && (nl) == 0)                     \
            {                                                              \
                _qh += (nh) + 1;                                           \
                _r = -_qh * (d);                                           \
                _mask = -(mp_limb_t)(_r > _ql); /* both > and >= are OK */ \
                _qh += _mask;                                              \
                _r += _mask & (d);                                         \
            }                                                              \
            else                                                           \
            {                                                              \
                add_ssaaaa(_qh, _ql, _qh, _ql, (nh) + 1, (nl));            \
                _r = (nl) - _qh * (d);                                     \
                _mask = -(mp_limb_t)(_r > _ql); /* both > and >= are OK */ \
                _qh += _mask;                                              \
                _r += _mask & (d);                                         \
                if (UNLIKELY(_r >= (d)))                                   \
                {                                                          \
                    _r -= (d);                                             \
                    _qh++;                                                 \
                }                                                          \
            }                                                              \
            (r) = _r;                                                      \
            (q) = _qh;                                                     \
        } while (0)

        /* Dividing (NH, NL) by D, returning the remainder only. Unlike
           udiv_qrnnd_preinv, works also for the case NH == D, where the
           quotient doesn't quite fit in a single limb. */
#define udiv_rnnd_preinv(r, nh, nl, d, di)                             \
        do                                                                 \
        {                                                                  \
            mp_limb_t _qh, _ql, _r, _mask;                                 \
            umul_ppmm(_qh, _ql, (nh), (di));                               \
            if (__builtin_constant_p(nl) && (nl) == 0)                     \
            {                                                              \
                _r = ~(_qh + (nh)) * (d);                                  \
                _mask = -(mp_limb_t)(_r > _ql); /* both > and >= are OK */ \
                _r += _mask & (d);                                         \
            }                                                              \
            else                                                           \
            {                                                              \
                add_ssaaaa(_qh, _ql, _qh, _ql, (nh) + 1, (nl));            \
                _r = (nl) - _qh * (d);                                     \
                _mask = -(mp_limb_t)(_r > _ql); /* both > and >= are OK */ \
                _r += _mask & (d);                                         \
                if (UNLIKELY(_r >= (d)))                                   \
                    _r -= (d);                                             \
            }                                                              \
            (r) = _r;                                                      \
        } while (0)

        /* Compute quotient the quotient and remainder for n / d. Requires d
           >= B^2 / 2 and n < d B. di is the inverse

             floor ((B^3 - 1) / (d0 + d1 B)) - B.

           NOTE: Output variables are updated multiple times. Only some inputs
           and outputs may overlap.
        */
#define udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv)             \
        do                                                                \
        {                                                                 \
            mp_limb_t _q0, _t1, _t0, _mask;                               \
            umul_ppmm((q), _q0, (n2), (dinv));                            \
            add_ssaaaa((q), _q0, (q), _q0, (n2), (n1));                   \
                                                                          \
            /* Compute the two most significant limbs of n - q'd */       \
            (r1) = (n1) - (d1) * (q);                                     \
            sub_ddmmss((r1), (r0), (r1), (n0), (d1), (d0));               \
            umul_ppmm(_t1, _t0, (d0), (q));                               \
            sub_ddmmss((r1), (r0), (r1), (r0), _t1, _t0);                 \
            (q)++;                                                        \
                                                                          \
            /* Conditionally adjust q and the remainders */               \
            _mask = -(mp_limb_t)((r1) >= _q0);                            \
            (q) += _mask;                                                 \
            add_ssaaaa((r1), (r0), (r1), (r0), _mask &(d1), _mask &(d0)); \
            if (UNLIKELY((r1) >= (d1)))                                   \
            {                                                             \
                if ((r1) > (d1) || (r0) >= (d0))                          \
                {                                                         \
                    (q)++;                                                \
                    sub_ddmmss((r1), (r0), (r1), (r0), (d1), (d0));       \
                }                                                         \
            }                                                             \
        } while (0)

#ifndef gpmpn_preinv_divrem_1 /* if not done with cpuvec in a fat binary */
#define gpmpn_preinv_divrem_1 __GPGMP_MPN(preinv_divrem_1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_preinv_divrem_1(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int);
#endif

        /* USE_PREINV_DIVREM_1 is whether to use gpmpn_preinv_divrem_1, as opposed to the
           plain gpmpn_divrem_1.  The default is yes, since the few CISC chips where
           preinv is not good have defines saying so.  */
#ifndef USE_PREINV_DIVREM_1
#define USE_PREINV_DIVREM_1 1
#endif

#if USE_PREINV_DIVREM_1
#define MPN_DIVREM_OR_PREINV_DIVREM_1(qp, xsize, ap, size, d, dinv, shift) \
        gpmpn_preinv_divrem_1(qp, xsize, ap, size, d, dinv, shift)
#else
#define MPN_DIVREM_OR_PREINV_DIVREM_1(qp, xsize, ap, size, d, dinv, shift) \
        gpmpn_divrem_1(qp, xsize, ap, size, d)
#endif

#ifndef PREINV_MOD_1_TO_MOD_1_THRESHOLD
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD 10
#endif

        /* This selection may seem backwards.  The reason gpmpn_mod_1 typically takes
           over for larger sizes is that it uses the mod_1_1 function.  */
#define MPN_MOD_OR_PREINV_MOD_1(src, size, divisor, inverse) \
        (BELOW_THRESHOLD(size, PREINV_MOD_1_TO_MOD_1_THRESHOLD)  \
             ? gpmpn_preinv_mod_1(src, size, divisor, inverse)     \
             : gpmpn_mod_1(src, size, divisor))
GPGMP_MPN_NAMESPACE_BEGIN
#ifndef gpmpn_mod_34lsub1 /* if not done with cpuvec in a fat binary */
#define gpmpn_mod_34lsub1 __GPGMP_MPN(mod_34lsub1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_34lsub1(mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;
#endif
GPGMP_MPN_NAMESPACE_END
        /* DIVEXACT_1_THRESHOLD is at what size to use gpmpn_divexact_1, as opposed to
           plain gpmpn_divrem_1.  Likewise BMOD_1_TO_MOD_1_THRESHOLD for
           gpmpn_modexact_1_odd against plain gpmpn_mod_1.  On most CPUs divexact and
           modexact are faster at all sizes, so the defaults are 0.  Those CPUs
           where this is not right have a tuned threshold.  */
#ifndef DIVEXACT_1_THRESHOLD
#define DIVEXACT_1_THRESHOLD 0
#endif
#ifndef BMOD_1_TO_MOD_1_THRESHOLD
#define BMOD_1_TO_MOD_1_THRESHOLD 10
#endif

#define MPN_DIVREM_OR_DIVEXACT_1(rp, up, n, d)                        \
        do                                                                \
        {                                                                 \
            if (BELOW_THRESHOLD(n, DIVEXACT_1_THRESHOLD))                 \
                ASSERT_NOCARRY(gpmpn_divrem_1(rp, (mp_size_t)0, up, n, d)); \
            else                                                          \
            {                                                             \
                ASSERT(gpmpn_mod_1(up, n, d) == 0);                         \
                gpmpn_divexact_1(rp, up, n, d);                             \
            }                                                             \
        } while (0)
GPGMP_MPN_NAMESPACE_BEGIN
#ifndef gpmpn_modexact_1c_odd /* if not done with cpuvec in a fat binary */
#define gpmpn_modexact_1c_odd __GPGMP_MPN(modexact_1c_odd)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_modexact_1c_odd(mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;
#endif

#if HAVE_NATIVE_gpmpn_modexact_1_odd
#define gpmpn_modexact_1_odd __GPGMP_MPN(modexact_1_odd)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_modexact_1_odd(mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;
#else
#define gpmpn_modexact_1_odd(src, size, divisor) \
        gpmpn_modexact_1c_odd(src, size, divisor, CNST_LIMB(0))
#endif
GPGMP_MPN_NAMESPACE_END
#define MPN_MOD_OR_MODEXACT_1_ODD(src, size, divisor) \
        (BELOW_THRESHOLD(size, BMOD_1_TO_MOD_1_THRESHOLD) \
             ? gpmpn_modexact_1_odd(src, size, divisor)     \
             : gpmpn_mod_1(src, size, divisor))

        /* binvert_limb() sets inv to the multiplicative inverse of n modulo
           2^GMP_NUMB_BITS, ie. satisfying inv*n == 1 mod 2^GMP_NUMB_BITS.
           n must be odd (otherwise such an inverse doesn't exist).

           This is not to be confused with invert_limb(), which is completely
           different.

           The table lookup gives an inverse with the low 8 bits valid, and each
           multiply step doubles the number of bits.  See Jebelean "An algorithm for
           exact division" end of section 4 (reference in gmp.texi).

           Possible enhancement: Could use UHWtype until the last step, if half-size
           multiplies are faster (might help under _LONG_LONG_LIMB).

           Alternative: As noted in Granlund and Montgomery "Division by Invariant
           Integers using Multiplication" (reference in gmp.texi), n itself gives a
           3-bit inverse immediately, and could be used instead of a table lookup.
           A 4-bit inverse can be obtained effectively from xoring bits 1 and 2 into
           bit 3, for instance with (((n + 2) & 4) << 1) ^ n.  */

#define binvert_limb_table __gmp_binvert_limb_table
#ifdef __CUDA_ARCH__
        __GPGMP_DECLSPEC __device__ const unsigned char binvert_limb_table[128] = {
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
            0x11, 0x3B, 0x5D, 0xC7, 0x49, 0x33, 0x55, 0xFF};
#else
__GPGMP_DECLSPEC const unsigned char binvert_limb_table[128] = {
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
    0x11, 0x3B, 0x5D, 0xC7, 0x49, 0x33, 0x55, 0xFF};
#endif

#define binvert_limb(inv, n)                                   \
        do                                                         \
        {                                                          \
            mp_limb_t __n = (n);                                   \
            mp_limb_t __inv;                                       \
            ASSERT((__n & 1) == 1);                                \
                                                                   \
            __inv = binvert_limb_table[(__n / 2) & 0x7F]; /*  8 */ \
            if (GMP_NUMB_BITS > 8)                                 \
                __inv = 2 * __inv - __inv * __inv * __n;           \
            if (GMP_NUMB_BITS > 16)                                \
                __inv = 2 * __inv - __inv * __inv * __n;           \
            if (GMP_NUMB_BITS > 32)                                \
                __inv = 2 * __inv - __inv * __inv * __n;           \
                                                                   \
            if (GMP_NUMB_BITS > 64)                                \
            {                                                      \
                int __invbits = 64;                                \
                do                                                 \
                {                                                  \
                    __inv = 2 * __inv - __inv * __inv * __n;       \
                    __invbits *= 2;                                \
                } while (__invbits < GMP_NUMB_BITS);               \
            }                                                      \
                                                                   \
            ASSERT((__inv * __n & GMP_NUMB_MASK) == 1);            \
            (inv) = __inv & GMP_NUMB_MASK;                         \
        } while (0)
#define modlimb_invert binvert_limb /* backward compatibility */

        /* Multiplicative inverse of 3, modulo 2^GMP_NUMB_BITS.
           Eg. 0xAAAAAAAB for 32 bits, 0xAAAAAAAAAAAAAAAB for 64 bits.
           GMP_NUMB_MAX/3*2+1 is right when GMP_NUMB_BITS is even, but when it's odd
           we need to start from GMP_NUMB_MAX>>1. */
#define MODLIMB_INVERSE_3 (((GMP_NUMB_MAX >> (GMP_NUMB_BITS % 2)) / 3) * 2 + 1)

        /* ceil(GMP_NUMB_MAX/3) and ceil(2*GMP_NUMB_MAX/3).
           These expressions work because GMP_NUMB_MAX%3 != 0 for all GMP_NUMB_BITS. */
#define GMP_NUMB_CEIL_MAX_DIV3 (GMP_NUMB_MAX / 3 + 1)
#define GMP_NUMB_CEIL_2MAX_DIV3 ((GMP_NUMB_MAX >> 1) / 3 + 1 + GMP_NUMB_HIGHBIT)

        /* Set r to -a mod d.  a>=d is allowed.  Can give r>d.  All should be limbs.

           It's not clear whether this is the best way to do this calculation.
           Anything congruent to -a would be fine for the one limb congruence
           tests.  */

#define NEG_MOD(r, a, d)                                          \
        do                                                            \
        {                                                             \
            ASSERT((d) != 0);                                         \
            ASSERT_LIMB(a);                                           \
            ASSERT_LIMB(d);                                           \
                                                                      \
            if ((a) <= (d))                                           \
            {                                                         \
                /* small a is reasonably likely */                    \
                (r) = (d) - (a);                                      \
            }                                                         \
            else                                                      \
            {                                                         \
                unsigned __twos;                                      \
                mp_limb_t __dnorm;                                    \
                count_leading_zeros(__twos, d);                       \
                __twos -= GMP_NAIL_BITS;                              \
                __dnorm = (d) << __twos;                              \
                (r) = ((a) <= __dnorm ? __dnorm : 2 * __dnorm) - (a); \
            }                                                         \
                                                                      \
            ASSERT_LIMB(r);                                           \
        } while (0)

        /* A bit mask of all the least significant zero bits of n, or -1 if n==0. */
#define LOW_ZEROS_MASK(n) (((n) & -(n)) - 1)

        /* ULONG_PARITY sets "p" to 1 if there's an odd number of 1 bits in "n", or
           to 0 if there's an even number.  "n" should be an unsigned long and "p"
           an int.  */

#if defined(__GNUC__) && !defined(NO_ASM) && HAVE_HOST_CPU_alpha_CIX
#define ULONG_PARITY(p, n)                            \
        do                                                \
        {                                                 \
            int __p;                                      \
            __asm__("ctpop %1, %0" : "=r"(__p) : "r"(n)); \
            (p) = __p & 1;                                \
        } while (0)
#endif

        /* Cray intrinsic _popcnt. */
#ifdef _CRAY
#define ULONG_PARITY(p, n)    \
        do                        \
        {                         \
            (p) = _popcnt(n) & 1; \
        } while (0)
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(NO_ASM) && defined(__ia64)
        /* unsigned long is either 32 or 64 bits depending on the ABI, zero extend
           to a 64 bit unsigned long long for popcnt */
#define ULONG_PARITY(p, n)                                \
        do                                                    \
        {                                                     \
            unsigned long long __n = (unsigned long)(n);      \
            int __p;                                          \
            __asm__("popcnt %0 = %1" : "=r"(__p) : "r"(__n)); \
            (p) = __p & 1;                                    \
        } while (0)
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(NO_ASM) && HAVE_HOST_CPU_FAMILY_x86
#if __GMP_GNUC_PREREQ(3, 1)
#define __GMP_qm "=Qm"
#define __GMP_q "=Q"
#else
#define __GMP_qm "=qm"
#define __GMP_q "=q"
#endif
#define ULONG_PARITY(p, n)                    \
        do                                        \
        {                                         \
            char __p;                             \
            unsigned long __n = (n);              \
            __n ^= (__n >> 16);                   \
            __asm__("xorb %h1, %b1\n\t"           \
                    "setpo %0"                    \
                    : __GMP_qm(__p), __GMP_q(__n) \
                    : "1"(__n));                  \
            (p) = __p;                            \
        } while (0)
#endif

#if !defined(ULONG_PARITY)
#define ULONG_PARITY(p, n)                      \
        do                                          \
        {                                           \
            unsigned long __n = (n);                \
            int __p = 0;                            \
            do                                      \
            {                                       \
                __p ^= 0x96696996L >> (__n & 0x1F); \
                __n >>= 5;                          \
            } while (__n != 0);                     \
                                                    \
            (p) = __p & 1;                          \
        } while (0)
#endif

        /* 3 cycles on 604 or 750 since shifts and rlwimi's can pair.  gcc (as of
           version 3.1 at least) doesn't seem to know how to generate rlwimi for
           anything other than bit-fields, so use "asm".  */
#if defined(__GNUC__) && !defined(NO_ASM) && HAVE_HOST_CPU_FAMILY_powerpc && GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)                                      \
        do                                                            \
        {                                                             \
            mp_limb_t __bswapl_src = (src);                           \
            mp_limb_t __tmp1 = __bswapl_src >> 24; /* low byte */     \
            mp_limb_t __tmp2 = __bswapl_src << 24; /* high byte */    \
            __asm__("rlwimi %0, %2, 24, 16, 23"    /* 2nd low */      \
                    : "=r"(__tmp1) : "0"(__tmp1), "r"(__bswapl_src)); \
            __asm__("rlwimi %0, %2,  8,  8, 15" /* 3nd high */        \
                    : "=r"(__tmp2) : "0"(__tmp2), "r"(__bswapl_src)); \
            (dst) = __tmp1 | __tmp2; /* whole */                      \
        } while (0)
#endif

        /* bswap is available on i486 and up and is fast.  A combination rorw $8 /
           roll $16 / rorw $8 is used in glibc for plain i386 (and in the linux
           kernel with xchgb instead of rorw), but this is not done here, because
           i386 means generic x86 and mixing word and dword operations will cause
           partial register stalls on P6 chips.  */
#if defined(__GNUC__) && !defined(NO_ASM) && HAVE_HOST_CPU_FAMILY_x86 && !HAVE_HOST_CPU_i386 && GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)                        \
        do                                              \
        {                                               \
            __asm__("bswap %0" : "=r"(dst) : "0"(src)); \
        } while (0)
#endif

#if defined(__GNUC__) && !defined(NO_ASM) && defined(__amd64__) && GMP_LIMB_BITS == 64
#define BSWAP_LIMB(dst, src)                         \
        do                                               \
        {                                                \
            __asm__("bswap %q0" : "=r"(dst) : "0"(src)); \
        } while (0)
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(NO_ASM) && defined(__ia64) && GMP_LIMB_BITS == 64
#define BSWAP_LIMB(dst, src)                                  \
        do                                                        \
        {                                                         \
            __asm__("mux1 %0 = %1, @rev" : "=r"(dst) : "r"(src)); \
        } while (0)
#endif

        /* As per glibc. */
#if defined(__GNUC__) && !defined(NO_ASM) && HAVE_HOST_CPU_FAMILY_m68k && GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)            \
        do                                  \
        {                                   \
            mp_limb_t __bswapl_src = (src); \
            __asm__("ror%.w %#8, %0\n\t"    \
                    "swap   %0\n\t"         \
                    "ror%.w %#8, %0"        \
                    : "=d"(dst)             \
                    : "0"(__bswapl_src));   \
        } while (0)
#endif

#if !defined(BSWAP_LIMB)
#if GMP_LIMB_BITS == 8
#define BSWAP_LIMB(dst, src) \
        do                       \
        {                        \
            (dst) = (src);       \
        } while (0)
#endif
#if GMP_LIMB_BITS == 16
#define BSWAP_LIMB(dst, src)                 \
        do                                       \
        {                                        \
            (dst) = ((src) << 8) + ((src) >> 8); \
        } while (0)
#endif
#if GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)                                                                   \
        do                                                                                         \
        {                                                                                          \
            (dst) =                                                                                \
                ((src) << 24) + (((src) & 0xFF00) << 8) + (((src) >> 8) & 0xFF00) + ((src) >> 24); \
        } while (0)
#endif
#if GMP_LIMB_BITS == 64
#define BSWAP_LIMB(dst, src)                                                                                                                                                                                           \
        do                                                                                                                                                                                                                 \
        {                                                                                                                                                                                                                  \
            (dst) =                                                                                                                                                                                                        \
                ((src) << 56) + (((src) & 0xFF00) << 40) + (((src) & 0xFF0000) << 24) + (((src) & 0xFF000000) << 8) + (((src) >> 8) & 0xFF000000) + (((src) >> 24) & 0xFF0000) + (((src) >> 40) & 0xFF00) + ((src) >> 56); \
        } while (0)
#endif
#endif

#if !defined(BSWAP_LIMB)
#define BSWAP_LIMB(dst, src)                                \
        do                                                      \
        {                                                       \
            mp_limb_t __bswapl_src = (src);                     \
            mp_limb_t __dstl = 0;                               \
            int __i;                                            \
            for (__i = 0; __i < GMP_LIMB_BYTES; __i++)          \
            {                                                   \
                __dstl = (__dstl << 8) | (__bswapl_src & 0xFF); \
                __bswapl_src >>= 8;                             \
            }                                                   \
            (dst) = __dstl;                                     \
        } while (0)
#endif

        /* Apparently lwbrx might be slow on some PowerPC chips, so restrict it to
           those we know are fast.  */
#if defined(__GNUC__) && !defined(NO_ASM) && GMP_LIMB_BITS == 32 && HAVE_LIMB_BIG_ENDIAN && (HAVE_HOST_CPU_powerpc604 || HAVE_HOST_CPU_powerpc604e || HAVE_HOST_CPU_powerpc750 || HAVE_HOST_CPU_powerpc7400)
#define BSWAP_LIMB_FETCH(limb, src)  \
        do                               \
        {                                \
            mp_srcptr __blf_src = (src); \
            mp_limb_t __limb;            \
            __asm__("lwbrx %0, 0, %1"    \
                    : "=r"(__limb)       \
                    : "r"(__blf_src),    \
                      "m"(*__blf_src));  \
            (limb) = __limb;             \
        } while (0)
#endif

#if !defined(BSWAP_LIMB_FETCH)
#define BSWAP_LIMB_FETCH(limb, src) BSWAP_LIMB(limb, *(src))
#endif

        /* On the same basis that lwbrx might be slow, restrict stwbrx to those we
           know are fast.  FIXME: Is this necessary?  */
#if defined(__GNUC__) && !defined(NO_ASM) && GMP_LIMB_BITS == 32 && HAVE_LIMB_BIG_ENDIAN && (HAVE_HOST_CPU_powerpc604 || HAVE_HOST_CPU_powerpc604e || HAVE_HOST_CPU_powerpc750 || HAVE_HOST_CPU_powerpc7400)
#define BSWAP_LIMB_STORE(dst, limb) \
        do                              \
        {                               \
            mp_ptr __dst = (dst);       \
            mp_limb_t __limb = (limb);  \
            __asm__("stwbrx %1, 0, %2"  \
                    : "=m"(*__dst)      \
                    : "r"(__limb),      \
                      "r"(__dst));      \
        } while (0)
#endif

#if !defined(BSWAP_LIMB_STORE)
#define BSWAP_LIMB_STORE(dst, limb) BSWAP_LIMB(*(dst), limb)
#endif

        /* Byte swap limbs from {src,size} and store at {dst,size}. */
#define MPN_BSWAP(dst, src, size)                       \
        do                                                  \
        {                                                   \
            mp_ptr __dst = (dst);                           \
            mp_srcptr __src = (src);                        \
            mp_size_t __size = (size);                      \
            mp_size_t __i;                                  \
            ASSERT((size) >= 0);                            \
            ASSERT(MPN_SAME_OR_SEPARATE_P(dst, src, size)); \
            CRAY_Pragma("_CRI ivdep");                      \
            for (__i = 0; __i < __size; __i++)              \
            {                                               \
                BSWAP_LIMB_FETCH(*__dst, __src);            \
                __dst++;                                    \
                __src++;                                    \
            }                                               \
        } while (0)

        /* Byte swap limbs from {dst,size} and store in reverse order at {src,size}. */
#define MPN_BSWAP_REVERSE(dst, src, size)             \
        do                                                \
        {                                                 \
            mp_ptr __dst = (dst);                         \
            mp_size_t __size = (size);                    \
            mp_srcptr __src = (src) + __size - 1;         \
            mp_size_t __i;                                \
            ASSERT((size) >= 0);                          \
            ASSERT(!MPN_OVERLAP_P(dst, size, src, size)); \
            CRAY_Pragma("_CRI ivdep");                    \
            for (__i = 0; __i < __size; __i++)            \
            {                                             \
                BSWAP_LIMB_FETCH(*__dst, __src);          \
                __dst++;                                  \
                __src--;                                  \
            }                                             \
        } while (0)

        /* No processor claiming to be SPARC v9 compliant seems to
           implement the POPC instruction.  Disable pattern for now.  */
#if 0
#if defined __GNUC__ && defined __sparc_v9__ && GMP_LIMB_BITS == 64
#define popc_limb(result, input)                            \
        do                                                      \
        {                                                       \
            DItype __res;                                       \
            __asm__("popc %1,%0" : "=r"(result) : "rI"(input)); \
        } while (0)
#endif
#endif

#if defined(__GNUC__) && !defined(NO_ASM) && HAVE_HOST_CPU_alpha_CIX
#define popc_limb(result, input)                             \
        do                                                       \
        {                                                        \
            __asm__("ctpop %1, %0" : "=r"(result) : "r"(input)); \
        } while (0)
#endif

        /* Cray intrinsic. */
#ifdef _CRAY
#define popc_limb(result, input)   \
        do                             \
        {                              \
            (result) = _popcnt(input); \
        } while (0)
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(NO_ASM) && defined(__ia64) && GMP_LIMB_BITS == 64
#define popc_limb(result, input)                               \
        do                                                         \
        {                                                          \
            __asm__("popcnt %0 = %1" : "=r"(result) : "r"(input)); \
        } while (0)
#endif

        /* Cool population count of an mp_limb_t.
           You have to figure out how this works, We won't tell you!

           The constants could also be expressed as:
             0x55... = [2^N / 3]     = [(2^N-1)/3]
             0x33... = [2^N / 5]     = [(2^N-1)/5]
             0x0f... = [2^N / 17]    = [(2^N-1)/17]
             (N is GMP_LIMB_BITS, [] denotes truncation.) */

#if !defined(popc_limb) && GMP_LIMB_BITS == 8
#define popc_limb(result, input)                                            \
        do                                                                      \
        {                                                                       \
            mp_limb_t __x = (input);                                            \
            __x -= (__x >> 1) & MP_LIMB_T_MAX / 3;                              \
            __x = ((__x >> 2) & MP_LIMB_T_MAX / 5) + (__x & MP_LIMB_T_MAX / 5); \
            __x = ((__x >> 4) + __x);                                           \
            (result) = __x & 0x0f;                                              \
        } while (0)
#endif

#if !defined(popc_limb) && GMP_LIMB_BITS == 16
#define popc_limb(result, input)                                                  \
        do                                                                            \
        {                                                                             \
            mp_limb_t __x = (input);                                                  \
            __x -= (__x >> 1) & MP_LIMB_T_MAX / 3;                                    \
            __x = ((__x >> 2) & MP_LIMB_T_MAX / 5) + (__x & MP_LIMB_T_MAX / 5);       \
            __x += (__x >> 4);                                                        \
            __x = ((__x >> 8) & MP_LIMB_T_MAX / 4369) + (__x & MP_LIMB_T_MAX / 4369); \
            (result) = __x;                                                           \
        } while (0)
#endif

#if !defined(popc_limb) && GMP_LIMB_BITS == 32
#define popc_limb(result, input)                                            \
        do                                                                      \
        {                                                                       \
            mp_limb_t __x = (input);                                            \
            __x -= (__x >> 1) & MP_LIMB_T_MAX / 3;                              \
            __x = ((__x >> 2) & MP_LIMB_T_MAX / 5) + (__x & MP_LIMB_T_MAX / 5); \
            __x = ((__x >> 4) + __x) & MP_LIMB_T_MAX / 17;                      \
            __x = ((__x >> 8) + __x);                                           \
            __x = ((__x >> 16) + __x);                                          \
            (result) = __x & 0xff;                                              \
        } while (0)
#endif

#if !defined(popc_limb) && GMP_LIMB_BITS == 64
#define popc_limb(result, input)                                            \
        do                                                                      \
        {                                                                       \
            mp_limb_t __x = (input);                                            \
            __x -= (__x >> 1) & MP_LIMB_T_MAX / 3;                              \
            __x = ((__x >> 2) & MP_LIMB_T_MAX / 5) + (__x & MP_LIMB_T_MAX / 5); \
            __x = ((__x >> 4) + __x) & MP_LIMB_T_MAX / 17;                      \
            __x = ((__x >> 8) + __x);                                           \
            __x = ((__x >> 16) + __x);                                          \
            __x = ((__x >> 32) + __x);                                          \
            (result) = __x & 0xff;                                              \
        } while (0)
#endif

        /* Define stuff for longlong.h.  */
#if HAVE_ATTRIBUTE_MODE
        typedef unsigned int UQItype __attribute__((mode(QI)));
        typedef int SItype __attribute__((mode(SI)));
        typedef unsigned int USItype __attribute__((mode(SI)));
        typedef int DItype __attribute__((mode(DI)));
        typedef unsigned int UDItype __attribute__((mode(DI)));
#else
typedef unsigned char UQItype;
typedef long SItype;
typedef unsigned long USItype;
#if HAVE_LONG_LONG
typedef long long int DItype;
typedef unsigned long long int UDItype;
#else /* Assume `long' gives us a wide enough type.  Needed for hppa2.0w.  */
typedef long int DItype;
typedef unsigned long int UDItype;
#endif
#endif

        typedef mp_limb_t UWtype;
        typedef unsigned int UHWtype;
#define W_TYPE_SIZE GMP_LIMB_BITS

        /* Define ieee_double_extract and _GMP_IEEE_FLOATS.

           Bit field packing is "implementation defined" according to C99, which
           leaves us at the compiler's mercy here.  For some systems packing is
           defined in the ABI (eg. x86).  In any case so far it seems universal that
           little endian systems pack from low to high, and big endian from high to
           low within the given type.

           Within the fields we rely on the integer endianness being the same as the
           float endianness, this is true everywhere we know of and it'd be a fairly
           strange system that did anything else.  */

#if HAVE_DOUBLE_IEEE_LITTLE_SWAPPED
#define _GMP_IEEE_FLOATS 1
        union ieee_double_extract
        {
            struct
            {
                gmp_uint_least32_t manh : 20;
                gmp_uint_least32_t exp : 11;
                gmp_uint_least32_t sig : 1;
                gmp_uint_least32_t manl : 32;
            } s;
            double d;
        };
#endif

#if HAVE_DOUBLE_IEEE_LITTLE_ENDIAN
#define _GMP_IEEE_FLOATS 1
        union ieee_double_extract
        {
            struct
            {
                gmp_uint_least32_t manl : 32;
                gmp_uint_least32_t manh : 20;
                gmp_uint_least32_t exp : 11;
                gmp_uint_least32_t sig : 1;
            } s;
            double d;
        };
#endif

#if HAVE_DOUBLE_IEEE_BIG_ENDIAN
#define _GMP_IEEE_FLOATS 1
        union ieee_double_extract
        {
            struct
            {
                gmp_uint_least32_t sig : 1;
                gmp_uint_least32_t exp : 11;
                gmp_uint_least32_t manh : 20;
                gmp_uint_least32_t manl : 32;
            } s;
            double d;
        };
#endif

#if HAVE_DOUBLE_VAX_D
        union double_extract
        {
            struct
            {
                gmp_uint_least32_t man3 : 7; /* highest 7 bits */
                gmp_uint_least32_t exp : 8;  /* excess-128 exponent */
                gmp_uint_least32_t sig : 1;
                gmp_uint_least32_t man2 : 16;
                gmp_uint_least32_t man1 : 16;
                gmp_uint_least32_t man0 : 16; /* lowest 16 bits */
            } s;
            double d;
        };
#endif

        /* Use (4.0 * ...) instead of (2.0 * ...) to work around buggy compilers
           that don't convert ulong->double correctly (eg. SunOS 4 native cc).  */
#define MP_BASE_AS_DOUBLE (4.0 * ((mp_limb_t)1 << (GMP_NUMB_BITS - 2)))
        /* Maximum number of limbs it will take to store any `double'.
           We assume doubles have 53 mantissa bits.  */
#define LIMBS_PER_DOUBLE ((53 + GMP_NUMB_BITS - 2) / GMP_NUMB_BITS + 1)
GPGMP_MPN_NAMESPACE_BEGIN
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_extract_double(mp_ptr, double);

#define gpmpn_get_d __ggpmpn_get_d
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE double gpmpn_get_d(mp_srcptr, mp_size_t, mp_size_t, long) __GMP_ATTRIBUTE_PURE;
GPGMP_MPN_NAMESPACE_END
        /* DOUBLE_NAN_INF_ACTION executes code a_nan if x is a NaN, or executes
           a_inf if x is an infinity.  Both are considered unlikely values, for
           branch prediction.  */

#if _GMP_IEEE_FLOATS
#define DOUBLE_NAN_INF_ACTION(x, a_nan, a_inf)  \
        do                                          \
        {                                           \
            union ieee_double_extract u;            \
            u.d = (x);                              \
            if (UNLIKELY(u.s.exp == 0x7FF))         \
            {                                       \
                if (u.s.manl == 0 && u.s.manh == 0) \
                {                                   \
                    a_inf;                          \
                }                                   \
                else                                \
                {                                   \
                    a_nan;                          \
                }                                   \
            }                                       \
        } while (0)
#endif

#if HAVE_DOUBLE_VAX_D || HAVE_DOUBLE_VAX_G || HAVE_DOUBLE_CRAY_CFP
        /* no nans or infs in these formats */
#define DOUBLE_NAN_INF_ACTION(x, a_nan, a_inf) \
        do                                         \
        {                                          \
        } while (0)
#endif

#ifndef DOUBLE_NAN_INF_ACTION
        /* Unknown format, try something generic.
           NaN should be "unordered", so x!=x.
           Inf should be bigger than DBL_MAX.  */
#define DOUBLE_NAN_INF_ACTION(x, a_nan, a_inf)                  \
        do                                                          \
        {                                                           \
            {                                                       \
                if (UNLIKELY((x) != (x)))                           \
                {                                                   \
                    a_nan;                                          \
                }                                                   \
                else if (UNLIKELY((x) > DBL_MAX || (x) < -DBL_MAX)) \
                {                                                   \
                    a_inf;                                          \
                }                                                   \
            }                                                       \
        } while (0)
#endif

        /* On m68k, x86 and amd64, gcc (and maybe other compilers) can hold doubles
           in the coprocessor, which means a bigger exponent range than normal, and
           depending on the rounding mode, a bigger mantissa than normal.  (See
           "Disappointments" in the gcc manual.)  FORCE_DOUBLE stores and fetches
           "d" through memory to force any rounding and overflows to occur.

           On amd64, and on x86s with SSE2, gcc (depending on options) uses the xmm
           registers, where there's no such extra precision and no need for the
           FORCE_DOUBLE.  We don't bother to detect this since the present uses for
           FORCE_DOUBLE are only in test programs and default generic C code.

           Not quite sure that an "automatic volatile" will use memory, but it does
           in gcc.  An asm("":"=m"(d):"0"(d)) can't be used to trick gcc, since
           apparently matching operands like "0" are only allowed on a register
           output.  gcc 3.4 warns about this, though in fact it and past versions
           seem to put the operand through memory as hoped.  */

#if (HAVE_HOST_CPU_FAMILY_m68k || HAVE_HOST_CPU_FAMILY_x86 || defined(__amd64__))
#define FORCE_DOUBLE(d)                    \
        do                                     \
        {                                      \
            volatile double __gmp_force = (d); \
            (d) = __gmp_force;                 \
        } while (0)
#else
#define FORCE_DOUBLE(d) \
        do                  \
        {                   \
        } while (0)
#endif

#define X 0xff
        __GPGMP_DECLSPEC const unsigned char __gmp_digit_value_tab[] =
            {
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, X, X, X, X, X, X,
                X, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, X, X, X, X, X,
                X, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, X, X, X, X, X, X,
                X, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, X, X, X, X, X,
                X, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
                X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X};
#undef X

        #ifdef __CUDA_ARCH__
        __GPGMP_DECLSPEC __device__ extern int __gpgmp_junk;
        __GPGMP_DECLSPEC __device__ const int __gpgmp_0 = 0;
        __GPGMP_DECLSPEC __device__ extern int gpgmp_errno;
        #else
        __GPGMP_DECLSPEC extern int gpgmp_errno;
        __GPGMP_DECLSPEC extern int __gpgmp_junk;
        __GPGMP_DECLSPEC const int __gpgmp_0 = 0;
        #endif


        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_exception(int);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_divide_by_zero(void);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_sqrt_of_negative(void);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_overflow_in_mpz(void);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_invalid_operation(void);

#define GMP_ERROR(code) __gpgmp_exception(code)
#define DIVIDE_BY_ZERO __gpgmp_divide_by_zero()
#define SQRT_OF_NEGATIVE __gpgmp_sqrt_of_negative()
#define MPZ_OVERFLOW __gpgmp_overflow_in_mpz()

#if defined _LONG_LONG_LIMB
#define CNST_LIMB(C) ((mp_limb_t)C##LL)
#else /* not _LONG_LONG_LIMB */
#define CNST_LIMB(C) ((mp_limb_t)C##L)
#endif /* _LONG_LONG_LIMB */

#ifdef __CUDA_ARCH__
        __GPGMP_DECLSPEC __device__ const struct bases mp_bases[257] =
            {
                /*   0 */ {0, 0, 0, 0, 0},
                /*   1 */ {0, 0, 0, 0, 0},
                /*   2 */ {64, CNST_LIMB(0xffffffffffffffff), CNST_LIMB(0x1fffffffffffffff), CNST_LIMB(0x1), CNST_LIMB(0x0)},
                /*   3 */ {40, CNST_LIMB(0xa1849cc1a9a9e94e), CNST_LIMB(0x32b803473f7ad0f3), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
                /*   4 */ {32, CNST_LIMB(0x7fffffffffffffff), CNST_LIMB(0x3fffffffffffffff), CNST_LIMB(0x2), CNST_LIMB(0x0)},
                /*   5 */ {27, CNST_LIMB(0x6e40d1a4143dcb94), CNST_LIMB(0x4a4d3c25e68dc57f), CNST_LIMB(0x6765c793fa10079d), CNST_LIMB(0x3ce9a36f23c0fc90)},
                /*   6 */ {24, CNST_LIMB(0x6308c91b702a7cf4), CNST_LIMB(0x52b803473f7ad0f3), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295)},
                /*   7 */ {22, CNST_LIMB(0x5b3064eb3aa6d388), CNST_LIMB(0x59d5d9fd5010b366), CNST_LIMB(0x3642798750226111), CNST_LIMB(0x2df495ccaa57147b)},
                /*   8 */ {21, CNST_LIMB(0x5555555555555555), CNST_LIMB(0x5fffffffffffffff), CNST_LIMB(0x3), CNST_LIMB(0x0)},
                /*   9 */ {20, CNST_LIMB(0x50c24e60d4d4f4a7), CNST_LIMB(0x6570068e7ef5a1e7), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
                /*  10 */ {19, CNST_LIMB(0x4d104d427de7fbcc), CNST_LIMB(0x6a4d3c25e68dc57f), CNST_LIMB(0x8ac7230489e80000), CNST_LIMB(0xd83c94fb6d2ac34a)},
                /*  11 */ {18, CNST_LIMB(0x4a00270775914e88), CNST_LIMB(0x6eb3a9f01975077f), CNST_LIMB(0x4d28cb56c33fa539), CNST_LIMB(0xa8adf7ae45e7577b)},
                /*  12 */ {17, CNST_LIMB(0x4768ce0d05818e12), CNST_LIMB(0x72b803473f7ad0f3), CNST_LIMB(0x1eca170c00000000), CNST_LIMB(0xa10c2bec5da8f8f)},
                /*  13 */ {17, CNST_LIMB(0x452e53e365907bda), CNST_LIMB(0x766a008e4788cbcd), CNST_LIMB(0x780c7372621bd74d), CNST_LIMB(0x10f4becafe412ec3)},
                /*  14 */ {16, CNST_LIMB(0x433cfffb4b5aae55), CNST_LIMB(0x79d5d9fd5010b366), CNST_LIMB(0x1e39a5057d810000), CNST_LIMB(0xf08480f672b4e86)},
                /*  15 */ {16, CNST_LIMB(0x41867711b4f85355), CNST_LIMB(0x7d053f6d26089673), CNST_LIMB(0x5b27ac993df97701), CNST_LIMB(0x6779c7f90dc42f48)},
                /*  16 */ {16, CNST_LIMB(0x3fffffffffffffff), CNST_LIMB(0x7fffffffffffffff), CNST_LIMB(0x4), CNST_LIMB(0x0)},
                /*  17 */ {15, CNST_LIMB(0x3ea16afd58b10966), CNST_LIMB(0x82cc7edf592262cf), CNST_LIMB(0x27b95e997e21d9f1), CNST_LIMB(0x9c71e11bab279323)},
                /*  18 */ {15, CNST_LIMB(0x3d64598d154dc4de), CNST_LIMB(0x8570068e7ef5a1e7), CNST_LIMB(0x5da0e1e53c5c8000), CNST_LIMB(0x5dfaa697ec6f6a1c)},
                /*  19 */ {15, CNST_LIMB(0x3c43c23018bb5563), CNST_LIMB(0x87ef05ae409a0288), CNST_LIMB(0xd2ae3299c1c4aedb), CNST_LIMB(0x3711783f6be7e9ec)},
                /*  20 */ {14, CNST_LIMB(0x3b3b9a42873069c7), CNST_LIMB(0x8a4d3c25e68dc57f), CNST_LIMB(0x16bcc41e90000000), CNST_LIMB(0x6849b86a12b9b01e)},
                /*  21 */ {14, CNST_LIMB(0x3a4898f06cf41ac9), CNST_LIMB(0x8c8ddd448f8b845a), CNST_LIMB(0x2d04b7fdd9c0ef49), CNST_LIMB(0x6bf097ba5ca5e239)},
                /*  22 */ {14, CNST_LIMB(0x39680b13582e7c18), CNST_LIMB(0x8eb3a9f01975077f), CNST_LIMB(0x5658597bcaa24000), CNST_LIMB(0x7b8015c8d7af8f08)},
                /*  23 */ {14, CNST_LIMB(0x3897b2b751ae561a), CNST_LIMB(0x90c10500d63aa658), CNST_LIMB(0xa0e2073737609371), CNST_LIMB(0x975a24b3a3151b38)},
                /*  24 */ {13, CNST_LIMB(0x37d5aed131f19c98), CNST_LIMB(0x92b803473f7ad0f3), CNST_LIMB(0xc29e98000000000), CNST_LIMB(0x50bd367972689db1)},
                /*  25 */ {13, CNST_LIMB(0x372068d20a1ee5ca), CNST_LIMB(0x949a784bcd1b8afe), CNST_LIMB(0x14adf4b7320334b9), CNST_LIMB(0x8c240c4aecb13bb5)},
                /*  26 */ {13, CNST_LIMB(0x3676867e5d60de29), CNST_LIMB(0x966a008e4788cbcd), CNST_LIMB(0x226ed36478bfa000), CNST_LIMB(0xdbd2e56854e118c9)},
                /*  27 */ {13, CNST_LIMB(0x35d6deeb388df86f), CNST_LIMB(0x982809d5be7072db), CNST_LIMB(0x383d9170b85ff80b), CNST_LIMB(0x2351ffcaa9c7c4ae)},
                /*  28 */ {13, CNST_LIMB(0x354071d61c77fa2e), CNST_LIMB(0x99d5d9fd5010b366), CNST_LIMB(0x5a3c23e39c000000), CNST_LIMB(0x6b24188ca33b0636)},
                /*  29 */ {13, CNST_LIMB(0x34b260c5671b18ac), CNST_LIMB(0x9b74948f5532da4b), CNST_LIMB(0x8e65137388122bcd), CNST_LIMB(0xcc3dceaf2b8ba99d)},
                /*  30 */ {13, CNST_LIMB(0x342be986572b45cc), CNST_LIMB(0x9d053f6d26089673), CNST_LIMB(0xdd41bb36d259e000), CNST_LIMB(0x2832e835c6c7d6b6)},
                /*  31 */ {12, CNST_LIMB(0x33ac61b998fbbdf2), CNST_LIMB(0x9e88c6b3626a72aa), CNST_LIMB(0xaee5720ee830681), CNST_LIMB(0x76b6aa272e1873c5)},
                /*  32 */ {12, CNST_LIMB(0x3333333333333333), CNST_LIMB(0x9fffffffffffffff), CNST_LIMB(0x5), CNST_LIMB(0x0)},
                /*  33 */ {12, CNST_LIMB(0x32bfd90114c12861), CNST_LIMB(0xa16bad3758efd873), CNST_LIMB(0x172588ad4f5f0981), CNST_LIMB(0x61eaf5d402c7bf4f)},
                /*  34 */ {12, CNST_LIMB(0x3251dcf6169e45f2), CNST_LIMB(0xa2cc7edf592262cf), CNST_LIMB(0x211e44f7d02c1000), CNST_LIMB(0xeeb658123ffb27ec)},
                /*  35 */ {12, CNST_LIMB(0x31e8d59f180dc630), CNST_LIMB(0xa4231623369e78e5), CNST_LIMB(0x2ee56725f06e5c71), CNST_LIMB(0x5d5e3762e6fdf509)},
                /*  36 */ {12, CNST_LIMB(0x3184648db8153e7a), CNST_LIMB(0xa570068e7ef5a1e7), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295)},
                /*  37 */ {12, CNST_LIMB(0x312434e89c35dacd), CNST_LIMB(0xa6b3d78b6d3b24fb), CNST_LIMB(0x5b5b57f8a98a5dd1), CNST_LIMB(0x66ae7831762efb6f)},
                /*  38 */ {12, CNST_LIMB(0x30c7fa349460a541), CNST_LIMB(0xa7ef05ae409a0288), CNST_LIMB(0x7dcff8986ea31000), CNST_LIMB(0x47388865a00f544)},
                /*  39 */ {12, CNST_LIMB(0x306f6f4c8432bc6d), CNST_LIMB(0xa92203d587039cc1), CNST_LIMB(0xabd4211662a6b2a1), CNST_LIMB(0x7d673c33a123b54c)},
                /*  40 */ {12, CNST_LIMB(0x301a557ffbfdd252), CNST_LIMB(0xaa4d3c25e68dc57f), CNST_LIMB(0xe8d4a51000000000), CNST_LIMB(0x19799812dea11197)},
                /*  41 */ {11, CNST_LIMB(0x2fc873d1fda55f3b), CNST_LIMB(0xab7110e6ce866f2b), CNST_LIMB(0x7a32956ad081b79), CNST_LIMB(0xc27e62e0686feae)},
                /*  42 */ {11, CNST_LIMB(0x2f799652a4e6dc49), CNST_LIMB(0xac8ddd448f8b845a), CNST_LIMB(0x9f49aaff0e86800), CNST_LIMB(0x9b6e7507064ce7c7)},
                /*  43 */ {11, CNST_LIMB(0x2f2d8d8f64460aad), CNST_LIMB(0xada3f5fb9c415052), CNST_LIMB(0xce583bb812d37b3), CNST_LIMB(0x3d9ac2bf66cfed94)},
                /*  44 */ {11, CNST_LIMB(0x2ee42e164e8f53a4), CNST_LIMB(0xaeb3a9f01975077f), CNST_LIMB(0x109b79a654c00000), CNST_LIMB(0xed46bc50ce59712a)},
                /*  45 */ {11, CNST_LIMB(0x2e9d500984041dbd), CNST_LIMB(0xafbd42b465836767), CNST_LIMB(0x1543beff214c8b95), CNST_LIMB(0x813d97e2c89b8d46)},
                /*  46 */ {11, CNST_LIMB(0x2e58cec05a6a8144), CNST_LIMB(0xb0c10500d63aa658), CNST_LIMB(0x1b149a79459a3800), CNST_LIMB(0x2e81751956af8083)},
                /*  47 */ {11, CNST_LIMB(0x2e1688743ef9104c), CNST_LIMB(0xb1bf311e95d00de3), CNST_LIMB(0x224edfb5434a830f), CNST_LIMB(0xdd8e0a95e30c0988)},
                /*  48 */ {11, CNST_LIMB(0x2dd65df7a583598f), CNST_LIMB(0xb2b803473f7ad0f3), CNST_LIMB(0x2b3fb00000000000), CNST_LIMB(0x7ad4dd48a0b5b167)},
                /*  49 */ {11, CNST_LIMB(0x2d9832759d5369c4), CNST_LIMB(0xb3abb3faa02166cc), CNST_LIMB(0x3642798750226111), CNST_LIMB(0x2df495ccaa57147b)},
                /*  50 */ {11, CNST_LIMB(0x2d5beb38dcd1394c), CNST_LIMB(0xb49a784bcd1b8afe), CNST_LIMB(0x43c33c1937564800), CNST_LIMB(0xe392010175ee5962)},
                /*  51 */ {11, CNST_LIMB(0x2d216f7943e2ba6a), CNST_LIMB(0xb5848226989d33c3), CNST_LIMB(0x54411b2441c3cd8b), CNST_LIMB(0x84eaf11b2fe7738e)},
                /*  52 */ {11, CNST_LIMB(0x2ce8a82efbb3ff2c), CNST_LIMB(0xb66a008e4788cbcd), CNST_LIMB(0x6851455acd400000), CNST_LIMB(0x3a1e3971e008995d)},
                /*  53 */ {11, CNST_LIMB(0x2cb17fea7ad7e332), CNST_LIMB(0xb74b1fd64e0753c6), CNST_LIMB(0x80a23b117c8feb6d), CNST_LIMB(0xfd7a462344ffce25)},
                /*  54 */ {11, CNST_LIMB(0x2c7be2b0cfa1ba50), CNST_LIMB(0xb82809d5be7072db), CNST_LIMB(0x9dff7d32d5dc1800), CNST_LIMB(0x9eca40b40ebcef8a)},
                /*  55 */ {11, CNST_LIMB(0x2c47bddba92d7463), CNST_LIMB(0xb900e6160002ccfe), CNST_LIMB(0xc155af6faeffe6a7), CNST_LIMB(0x52fa161a4a48e43d)},
                /*  56 */ {11, CNST_LIMB(0x2c14fffcaa8b131e), CNST_LIMB(0xb9d5d9fd5010b366), CNST_LIMB(0xebb7392e00000000), CNST_LIMB(0x1607a2cbacf930c1)},
                /*  57 */ {10, CNST_LIMB(0x2be398c3a38be053), CNST_LIMB(0xbaa708f58014d37c), CNST_LIMB(0x50633659656d971), CNST_LIMB(0x97a014f8e3be55f1)},
                /*  58 */ {10, CNST_LIMB(0x2bb378e758451068), CNST_LIMB(0xbb74948f5532da4b), CNST_LIMB(0x5fa8624c7fba400), CNST_LIMB(0x568df8b76cbf212c)},
                /*  59 */ {10, CNST_LIMB(0x2b8492108be5e5f7), CNST_LIMB(0xbc3e9ca2e1a05533), CNST_LIMB(0x717d9faa73c5679), CNST_LIMB(0x20ba7c4b4e6ef492)},
                /*  60 */ {10, CNST_LIMB(0x2b56d6c70d55481b), CNST_LIMB(0xbd053f6d26089673), CNST_LIMB(0x86430aac6100000), CNST_LIMB(0xe81ee46b9ef492f5)},
                /*  61 */ {10, CNST_LIMB(0x2b2a3a608c72ddd5), CNST_LIMB(0xbdc899ab3ff56c5e), CNST_LIMB(0x9e64d9944b57f29), CNST_LIMB(0x9dc0d10d51940416)},
                /*  62 */ {10, CNST_LIMB(0x2afeb0f1060c7e41), CNST_LIMB(0xbe88c6b3626a72aa), CNST_LIMB(0xba5ca5392cb0400), CNST_LIMB(0x5fa8ed2f450272a5)},
                /*  63 */ {10, CNST_LIMB(0x2ad42f3c9aca595c), CNST_LIMB(0xbf45e08bcf06554e), CNST_LIMB(0xdab2ce1d022cd81), CNST_LIMB(0x2ba9eb8c5e04e641)},
                /*  64 */ {10, CNST_LIMB(0x2aaaaaaaaaaaaaaa), CNST_LIMB(0xbfffffffffffffff), CNST_LIMB(0x6), CNST_LIMB(0x0)},
                /*  65 */ {10, CNST_LIMB(0x2a82193a13425883), CNST_LIMB(0xc0b73cb42e16914c), CNST_LIMB(0x12aeed5fd3e2d281), CNST_LIMB(0xb67759cc00287bf1)},
                /*  66 */ {10, CNST_LIMB(0x2a5a717672f66450), CNST_LIMB(0xc16bad3758efd873), CNST_LIMB(0x15c3da1572d50400), CNST_LIMB(0x78621feeb7f4ed33)},
                /*  67 */ {10, CNST_LIMB(0x2a33aa6e56d9c71c), CNST_LIMB(0xc21d6713f453f356), CNST_LIMB(0x194c05534f75ee29), CNST_LIMB(0x43d55b5f72943bc0)},
                /*  68 */ {10, CNST_LIMB(0x2a0dbbaa3bdfcea4), CNST_LIMB(0xc2cc7edf592262cf), CNST_LIMB(0x1d56299ada100000), CNST_LIMB(0x173decb64d1d4409)},
                /*  69 */ {10, CNST_LIMB(0x29e89d244eb4bfaf), CNST_LIMB(0xc379084815b5774c), CNST_LIMB(0x21f2a089a4ff4f79), CNST_LIMB(0xe29fb54fd6b6074f)},
                /*  70 */ {10, CNST_LIMB(0x29c44740d7db51e6), CNST_LIMB(0xc4231623369e78e5), CNST_LIMB(0x2733896c68d9a400), CNST_LIMB(0xa1f1f5c210d54e62)},
                /*  71 */ {10, CNST_LIMB(0x29a0b2c743b14d74), CNST_LIMB(0xc4caba789e2b8687), CNST_LIMB(0x2d2cf2c33b533c71), CNST_LIMB(0x6aac7f9bfafd57b2)},
                /*  72 */ {10, CNST_LIMB(0x297dd8dbb7c22a2d), CNST_LIMB(0xc570068e7ef5a1e7), CNST_LIMB(0x33f506e440000000), CNST_LIMB(0x3b563c2478b72ee2)},
                /*  73 */ {10, CNST_LIMB(0x295bb2f9285c8c1b), CNST_LIMB(0xc6130af40bc0ecbf), CNST_LIMB(0x3ba43bec1d062211), CNST_LIMB(0x12b536b574e92d1b)},
                /*  74 */ {10, CNST_LIMB(0x293a3aebe2be1c92), CNST_LIMB(0xc6b3d78b6d3b24fb), CNST_LIMB(0x4455872d8fd4e400), CNST_LIMB(0xdf86c03020404fa5)},
                /*  75 */ {10, CNST_LIMB(0x29196acc815ebd9f), CNST_LIMB(0xc7527b930c965bf2), CNST_LIMB(0x4e2694539f2f6c59), CNST_LIMB(0xa34adf02234eea8e)},
                /*  76 */ {10, CNST_LIMB(0x28f93cfb40f5c22a), CNST_LIMB(0xc7ef05ae409a0288), CNST_LIMB(0x5938006c18900000), CNST_LIMB(0x6f46eb8574eb59dd)},
                /*  77 */ {10, CNST_LIMB(0x28d9ac1badc64117), CNST_LIMB(0xc88983ed6985bae5), CNST_LIMB(0x65ad9912474aa649), CNST_LIMB(0x42459b481df47cec)},
                /*  78 */ {10, CNST_LIMB(0x28bab310a196b478), CNST_LIMB(0xc92203d587039cc1), CNST_LIMB(0x73ae9ff4241ec400), CNST_LIMB(0x1b424b95d80ca505)},
                /*  79 */ {10, CNST_LIMB(0x289c4cf88b774469), CNST_LIMB(0xc9b892675266f66c), CNST_LIMB(0x836612ee9c4ce1e1), CNST_LIMB(0xf2c1b982203a0dac)},
                /*  80 */ {10, CNST_LIMB(0x287e7529fb244e91), CNST_LIMB(0xca4d3c25e68dc57f), CNST_LIMB(0x9502f90000000000), CNST_LIMB(0xb7cdfd9d7bdbab7d)},
                /*  81 */ {10, CNST_LIMB(0x286127306a6a7a53), CNST_LIMB(0xcae00d1cfdeb43cf), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
                /*  82 */ {10, CNST_LIMB(0x28445ec93f792b1e), CNST_LIMB(0xcb7110e6ce866f2b), CNST_LIMB(0xbebf59a07dab4400), CNST_LIMB(0x57931eeaf85cf64f)},
                /*  83 */ {10, CNST_LIMB(0x282817e1038950fa), CNST_LIMB(0xcc0052b18b0e2a19), CNST_LIMB(0xd7540d4093bc3109), CNST_LIMB(0x305a944507c82f47)},
                /*  84 */ {10, CNST_LIMB(0x280c4e90c9ab1f45), CNST_LIMB(0xcc8ddd448f8b845a), CNST_LIMB(0xf2b96616f1900000), CNST_LIMB(0xe007ccc9c22781a)},
                /*  85 */ {9, CNST_LIMB(0x27f0ff1bc1ee87cd), CNST_LIMB(0xcd19bb053fb0284e), CNST_LIMB(0x336de62af2bca35), CNST_LIMB(0x3e92c42e000eeed4)},
                /*  86 */ {9, CNST_LIMB(0x27d625ecf571c340), CNST_LIMB(0xcda3f5fb9c415052), CNST_LIMB(0x39235ec33d49600), CNST_LIMB(0x1ebe59130db2795e)},
                /*  87 */ {9, CNST_LIMB(0x27bbbf95282fcd45), CNST_LIMB(0xce2c97d694adab3f), CNST_LIMB(0x3f674e539585a17), CNST_LIMB(0x268859e90f51b89)},
                /*  88 */ {9, CNST_LIMB(0x27a1c8c8ddaf84da), CNST_LIMB(0xceb3a9f01975077f), CNST_LIMB(0x4645b6958000000), CNST_LIMB(0xd24cde0463108cfa)},
                /*  89 */ {9, CNST_LIMB(0x27883e5e7df3f518), CNST_LIMB(0xcf393550f3aa6906), CNST_LIMB(0x4dcb74afbc49c19), CNST_LIMB(0xa536009f37adc383)},
                /*  90 */ {9, CNST_LIMB(0x276f1d4c9847e90e), CNST_LIMB(0xcfbd42b465836767), CNST_LIMB(0x56064e1d18d9a00), CNST_LIMB(0x7cea06ce1c9ace10)},
                /*  91 */ {9, CNST_LIMB(0x275662a841b30191), CNST_LIMB(0xd03fda8b97997f33), CNST_LIMB(0x5f04fe2cd8a39fb), CNST_LIMB(0x58db032e72e8ba43)},
                /*  92 */ {9, CNST_LIMB(0x273e0ba38d15a47b), CNST_LIMB(0xd0c10500d63aa658), CNST_LIMB(0x68d74421f5c0000), CNST_LIMB(0x388cc17cae105447)},
                /*  93 */ {9, CNST_LIMB(0x2726158c1b13cf03), CNST_LIMB(0xd140c9faa1e5439e), CNST_LIMB(0x738df1f6ab4827d), CNST_LIMB(0x1b92672857620ce0)},
                /*  94 */ {9, CNST_LIMB(0x270e7dc9c01d8e9b), CNST_LIMB(0xd1bf311e95d00de3), CNST_LIMB(0x7f3afbc9cfb5e00), CNST_LIMB(0x18c6a9575c2ade4)},
                /*  95 */ {9, CNST_LIMB(0x26f741dd3f070d61), CNST_LIMB(0xd23c41d42727c808), CNST_LIMB(0x8bf187fba88f35f), CNST_LIMB(0xd44da7da8e44b24f)},
                /*  96 */ {9, CNST_LIMB(0x26e05f5f16c2159e), CNST_LIMB(0xd2b803473f7ad0f3), CNST_LIMB(0x99c600000000000), CNST_LIMB(0xaa2f78f1b4cc6794)},
                /*  97 */ {9, CNST_LIMB(0x26c9d3fe61e80598), CNST_LIMB(0xd3327c6ab49ca6c8), CNST_LIMB(0xa8ce21eb6531361), CNST_LIMB(0x843c067d091ee4cc)},
                /*  98 */ {9, CNST_LIMB(0x26b39d7fc6ddab08), CNST_LIMB(0xd3abb3faa02166cc), CNST_LIMB(0xb92112c1a0b6200), CNST_LIMB(0x62005e1e913356e3)},
                /*  99 */ {9, CNST_LIMB(0x269db9bc7772a5cc), CNST_LIMB(0xd423b07e986aa967), CNST_LIMB(0xcad7718b8747c43), CNST_LIMB(0x4316eed01dedd518)},
                /* 100 */ {9, CNST_LIMB(0x268826a13ef3fde6), CNST_LIMB(0xd49a784bcd1b8afe), CNST_LIMB(0xde0b6b3a7640000), CNST_LIMB(0x2725dd1d243aba0e)},
                /* 101 */ {9, CNST_LIMB(0x2672e22d9dbdbd9f), CNST_LIMB(0xd510118708a8f8dd), CNST_LIMB(0xf2d8cf5fe6d74c5), CNST_LIMB(0xddd9057c24cb54f)},
                /* 102 */ {9, CNST_LIMB(0x265dea72f169cc99), CNST_LIMB(0xd5848226989d33c3), CNST_LIMB(0x1095d25bfa712600), CNST_LIMB(0xedeee175a736d2a1)},
                /* 103 */ {9, CNST_LIMB(0x26493d93a8cb2514), CNST_LIMB(0xd5f7cff41e09aeb8), CNST_LIMB(0x121b7c4c3698faa7), CNST_LIMB(0xc4699f3df8b6b328)},
                /* 104 */ {9, CNST_LIMB(0x2634d9c282f3ef82), CNST_LIMB(0xd66a008e4788cbcd), CNST_LIMB(0x13c09e8d68000000), CNST_LIMB(0x9ebbe7d859cb5a7c)},
                /* 105 */ {9, CNST_LIMB(0x2620bd41d8933adc), CNST_LIMB(0xd6db196a761949d9), CNST_LIMB(0x15876ccb0b709ca9), CNST_LIMB(0x7c828b9887eb2179)},
                /* 106 */ {9, CNST_LIMB(0x260ce662ef04088a), CNST_LIMB(0xd74b1fd64e0753c6), CNST_LIMB(0x17723c2976da2a00), CNST_LIMB(0x5d652ab99001adcf)},
                /* 107 */ {9, CNST_LIMB(0x25f95385547353fd), CNST_LIMB(0xd7ba18f93502e409), CNST_LIMB(0x198384e9c259048b), CNST_LIMB(0x4114f1754e5d7b32)},
                /* 108 */ {9, CNST_LIMB(0x25e60316448db8e1), CNST_LIMB(0xd82809d5be7072db), CNST_LIMB(0x1bbde41dfeec0000), CNST_LIMB(0x274b7c902f7e0188)},
                /* 109 */ {9, CNST_LIMB(0x25d2f390152f74f5), CNST_LIMB(0xd894f74b06ef8b40), CNST_LIMB(0x1e241d6e3337910d), CNST_LIMB(0xfc9e0fbb32e210c)},
                /* 110 */ {9, CNST_LIMB(0x25c02379aa9ad043), CNST_LIMB(0xd900e6160002ccfe), CNST_LIMB(0x20b91cee9901ee00), CNST_LIMB(0xf4afa3e594f8ea1f)},
                /* 111 */ {9, CNST_LIMB(0x25ad9165f2c18907), CNST_LIMB(0xd96bdad2acb5f5ef), CNST_LIMB(0x237ff9079863dfef), CNST_LIMB(0xcd85c32e9e4437b0)},
                /* 112 */ {9, CNST_LIMB(0x259b3bf36735c90c), CNST_LIMB(0xd9d5d9fd5010b366), CNST_LIMB(0x267bf47000000000), CNST_LIMB(0xa9bbb147e0dd92a8)},
                /* 113 */ {9, CNST_LIMB(0x258921cb955e7693), CNST_LIMB(0xda3ee7f38e181ed0), CNST_LIMB(0x29b08039fbeda7f1), CNST_LIMB(0x8900447b70e8eb82)},
                /* 114 */ {9, CNST_LIMB(0x257741a2ac9170af), CNST_LIMB(0xdaa708f58014d37c), CNST_LIMB(0x2d213df34f65f200), CNST_LIMB(0x6b0a92adaad5848a)},
                /* 115 */ {9, CNST_LIMB(0x25659a3711bc827d), CNST_LIMB(0xdb0e4126bcc86bd7), CNST_LIMB(0x30d201d957a7c2d3), CNST_LIMB(0x4f990ad8740f0ee5)},
                /* 116 */ {9, CNST_LIMB(0x25542a50f84b9c39), CNST_LIMB(0xdb74948f5532da4b), CNST_LIMB(0x34c6d52160f40000), CNST_LIMB(0x3670a9663a8d3610)},
                /* 117 */ {9, CNST_LIMB(0x2542f0c20000377d), CNST_LIMB(0xdbda071cc67e6db5), CNST_LIMB(0x3903f855d8f4c755), CNST_LIMB(0x1f5c44188057be3c)},
                /* 118 */ {9, CNST_LIMB(0x2531ec64d772bd64), CNST_LIMB(0xdc3e9ca2e1a05533), CNST_LIMB(0x3d8de5c8ec59b600), CNST_LIMB(0xa2bea956c4e4977)},
                /* 119 */ {9, CNST_LIMB(0x25211c1ce2fb5a6e), CNST_LIMB(0xdca258dca9331635), CNST_LIMB(0x4269541d1ff01337), CNST_LIMB(0xed68b23033c3637e)},
                /* 120 */ {9, CNST_LIMB(0x25107ed5e7c3ec3b), CNST_LIMB(0xdd053f6d26089673), CNST_LIMB(0x479b38e478000000), CNST_LIMB(0xc99cf624e50549c5)},
                /* 121 */ {9, CNST_LIMB(0x25001383bac8a744), CNST_LIMB(0xdd6753e032ea0efe), CNST_LIMB(0x4d28cb56c33fa539), CNST_LIMB(0xa8adf7ae45e7577b)},
                /* 122 */ {9, CNST_LIMB(0x24efd921f390bce3), CNST_LIMB(0xddc899ab3ff56c5e), CNST_LIMB(0x5317871fa13aba00), CNST_LIMB(0x8a5bc740b1c113e5)},
                /* 123 */ {9, CNST_LIMB(0x24dfceb3a26bb203), CNST_LIMB(0xde29142e0e01401f), CNST_LIMB(0x596d2f44de9fa71b), CNST_LIMB(0x6e6c7efb81cfbb9b)},
                /* 124 */ {9, CNST_LIMB(0x24cff3430a0341a7), CNST_LIMB(0xde88c6b3626a72aa), CNST_LIMB(0x602fd125c47c0000), CNST_LIMB(0x54aba5c5cada5f10)},
                /* 125 */ {9, CNST_LIMB(0x24c045e15c149931), CNST_LIMB(0xdee7b471b3a9507d), CNST_LIMB(0x6765c793fa10079d), CNST_LIMB(0x3ce9a36f23c0fc90)},
                /* 126 */ {9, CNST_LIMB(0x24b0c5a679267ae2), CNST_LIMB(0xdf45e08bcf06554e), CNST_LIMB(0x6f15be069b847e00), CNST_LIMB(0x26fb43de2c8cd2a8)},
                /* 127 */ {9, CNST_LIMB(0x24a171b0b31461c8), CNST_LIMB(0xdfa34e1177c23362), CNST_LIMB(0x7746b3e82a77047f), CNST_LIMB(0x12b94793db8486a1)},
                /* 128 */ {9, CNST_LIMB(0x2492492492492492), CNST_LIMB(0xdfffffffffffffff), CNST_LIMB(0x7), CNST_LIMB(0x0)},
                /* 129 */ {9, CNST_LIMB(0x24834b2c9d85cdfe), CNST_LIMB(0xe05bf942dbbc2145), CNST_LIMB(0x894953f7ea890481), CNST_LIMB(0xdd5deca404c0156d)},
                /* 130 */ {9, CNST_LIMB(0x247476f924137501), CNST_LIMB(0xe0b73cb42e16914c), CNST_LIMB(0x932abffea4848200), CNST_LIMB(0xbd51373330291de0)},
                /* 131 */ {9, CNST_LIMB(0x2465cbc00a40cec0), CNST_LIMB(0xe111cd1d5133412e), CNST_LIMB(0x9dacb687d3d6a163), CNST_LIMB(0x9fa4025d66f23085)},
                /* 132 */ {9, CNST_LIMB(0x245748bc980e0427), CNST_LIMB(0xe16bad3758efd873), CNST_LIMB(0xa8d8102a44840000), CNST_LIMB(0x842530ee2db4949d)},
                /* 133 */ {9, CNST_LIMB(0x2448ed2f49eb0633), CNST_LIMB(0xe1c4dfab90aab5ef), CNST_LIMB(0xb4b60f9d140541e5), CNST_LIMB(0x6aa7f2766b03dc25)},
                /* 134 */ {9, CNST_LIMB(0x243ab85da36e3167), CNST_LIMB(0xe21d6713f453f356), CNST_LIMB(0xc15065d4856e4600), CNST_LIMB(0x53035ba7ebf32e8d)},
                /* 135 */ {9, CNST_LIMB(0x242ca99203ea8c18), CNST_LIMB(0xe27545fba4fe385a), CNST_LIMB(0xceb1363f396d23c7), CNST_LIMB(0x3d12091fc9fb4914)},
                /* 136 */ {9, CNST_LIMB(0x241ec01b7cce4ea0), CNST_LIMB(0xe2cc7edf592262cf), CNST_LIMB(0xdce31b2488000000), CNST_LIMB(0x28b1cb81b1ef1849)},
                /* 137 */ {9, CNST_LIMB(0x2410fb4da9b3b0fc), CNST_LIMB(0xe323142dc8c66b55), CNST_LIMB(0xebf12a24bca135c9), CNST_LIMB(0x15c35be67ae3e2c9)},
                /* 138 */ {9, CNST_LIMB(0x24035a808a0f315e), CNST_LIMB(0xe379084815b5774c), CNST_LIMB(0xfbe6f8dbf88f4a00), CNST_LIMB(0x42a17bd09be1ff0)},
                /* 139 */ {8, CNST_LIMB(0x23f5dd105c67ab9d), CNST_LIMB(0xe3ce5d822ff4b643), CNST_LIMB(0x1ef156c084ce761), CNST_LIMB(0x8bf461f03cf0bbf)},
                /* 140 */ {8, CNST_LIMB(0x23e8825d7b05abb1), CNST_LIMB(0xe4231623369e78e5), CNST_LIMB(0x20c4e3b94a10000), CNST_LIMB(0xf3fbb43f68a32d05)},
                /* 141 */ {8, CNST_LIMB(0x23db49cc3a0866fe), CNST_LIMB(0xe4773465d54aded7), CNST_LIMB(0x22b0695a08ba421), CNST_LIMB(0xd84f44c48564dc19)},
                /* 142 */ {8, CNST_LIMB(0x23ce32c4c6cfb9f5), CNST_LIMB(0xe4caba789e2b8687), CNST_LIMB(0x24b4f35d7a4c100), CNST_LIMB(0xbe58ebcce7956abe)},
                /* 143 */ {8, CNST_LIMB(0x23c13cb308ab6ab7), CNST_LIMB(0xe51daa7e60fdd34c), CNST_LIMB(0x26d397284975781), CNST_LIMB(0xa5fac463c7c134b7)},
                /* 144 */ {8, CNST_LIMB(0x23b4670682c0c709), CNST_LIMB(0xe570068e7ef5a1e7), CNST_LIMB(0x290d74100000000), CNST_LIMB(0x8f19241e28c7d757)},
                /* 145 */ {8, CNST_LIMB(0x23a7b13237187c8b), CNST_LIMB(0xe5c1d0b53bc09fca), CNST_LIMB(0x2b63b3a37866081), CNST_LIMB(0x799a6d046c0ae1ae)},
                /* 146 */ {8, CNST_LIMB(0x239b1aac8ac74728), CNST_LIMB(0xe6130af40bc0ecbf), CNST_LIMB(0x2dd789f4d894100), CNST_LIMB(0x6566e37d746a9e40)},
                /* 147 */ {8, CNST_LIMB(0x238ea2ef2b24c379), CNST_LIMB(0xe663b741df9c37c0), CNST_LIMB(0x306a35e51b58721), CNST_LIMB(0x526887dbfb5f788f)},
                /* 148 */ {8, CNST_LIMB(0x23824976f4045a26), CNST_LIMB(0xe6b3d78b6d3b24fb), CNST_LIMB(0x331d01712e10000), CNST_LIMB(0x408af3382b8efd3d)},
                /* 149 */ {8, CNST_LIMB(0x23760dc3d6e4d729), CNST_LIMB(0xe7036db376537b90), CNST_LIMB(0x35f14200a827c61), CNST_LIMB(0x2fbb374806ec05f1)},
                /* 150 */ {8, CNST_LIMB(0x2369ef58c30bd43e), CNST_LIMB(0xe7527b930c965bf2), CNST_LIMB(0x38e858b62216100), CNST_LIMB(0x1fe7c0f0afce87fe)},
                /* 151 */ {8, CNST_LIMB(0x235dedbb8e82aa1c), CNST_LIMB(0xe7a102f9d39a9331), CNST_LIMB(0x3c03b2c13176a41), CNST_LIMB(0x11003d517540d32e)},
                /* 152 */ {8, CNST_LIMB(0x23520874dfeb1ffd), CNST_LIMB(0xe7ef05ae409a0288), CNST_LIMB(0x3f44c9b21000000), CNST_LIMB(0x2f5810f98eff0dc)},
                /* 153 */ {8, CNST_LIMB(0x23463f1019228dd7), CNST_LIMB(0xe83c856dd81804b7), CNST_LIMB(0x42ad23cef3113c1), CNST_LIMB(0xeb72e35e7840d910)},
                /* 154 */ {8, CNST_LIMB(0x233a911b42aa9b3c), CNST_LIMB(0xe88983ed6985bae5), CNST_LIMB(0x463e546b19a2100), CNST_LIMB(0xd27de19593dc3614)},
                /* 155 */ {8, CNST_LIMB(0x232efe26f7cf33f9), CNST_LIMB(0xe8d602d948f83829), CNST_LIMB(0x49f9fc3f96684e1), CNST_LIMB(0xbaf391fd3e5e6fc2)},
                /* 156 */ {8, CNST_LIMB(0x232385c65381b485), CNST_LIMB(0xe92203d587039cc1), CNST_LIMB(0x4de1c9c5dc10000), CNST_LIMB(0xa4bd38c55228c81d)},
                /* 157 */ {8, CNST_LIMB(0x2318278edde1b39b), CNST_LIMB(0xe96d887e26cd57b7), CNST_LIMB(0x51f77994116d2a1), CNST_LIMB(0x8fc5a8de8e1de782)},
                /* 158 */ {8, CNST_LIMB(0x230ce3187a6c2be9), CNST_LIMB(0xe9b892675266f66c), CNST_LIMB(0x563cd6bb3398100), CNST_LIMB(0x7bf9265bea9d3a3b)},
                /* 159 */ {8, CNST_LIMB(0x2301b7fd56ca21bb), CNST_LIMB(0xea03231d8d8224ba), CNST_LIMB(0x5ab3bb270beeb01), CNST_LIMB(0x69454b325983dccd)},
                /* 160 */ {8, CNST_LIMB(0x22f6a5d9da38341c), CNST_LIMB(0xea4d3c25e68dc57f), CNST_LIMB(0x5f5e10000000000), CNST_LIMB(0x5798ee2308c39df9)},
                /* 161 */ {8, CNST_LIMB(0x22ebac4c9580d89f), CNST_LIMB(0xea96defe264b59be), CNST_LIMB(0x643dce0ec16f501), CNST_LIMB(0x46e40ba0fa66a753)},
                /* 162 */ {8, CNST_LIMB(0x22e0caf633834beb), CNST_LIMB(0xeae00d1cfdeb43cf), CNST_LIMB(0x6954fe21e3e8100), CNST_LIMB(0x3717b0870b0db3a7)},
                /* 163 */ {8, CNST_LIMB(0x22d601796a418886), CNST_LIMB(0xeb28c7f233bdd372), CNST_LIMB(0x6ea5b9755f440a1), CNST_LIMB(0x2825e6775d11cdeb)},
                /* 164 */ {8, CNST_LIMB(0x22cb4f7aec6fd8b4), CNST_LIMB(0xeb7110e6ce866f2b), CNST_LIMB(0x74322a1c0410000), CNST_LIMB(0x1a01a1c09d1b4dac)},
                /* 165 */ {8, CNST_LIMB(0x22c0b4a15b80d83e), CNST_LIMB(0xebb8e95d3f7d9df2), CNST_LIMB(0x79fc8b6ae8a46e1), CNST_LIMB(0xc9eb0a8bebc8f3e)},
                /* 166 */ {8, CNST_LIMB(0x22b630953a28f77a), CNST_LIMB(0xec0052b18b0e2a19), CNST_LIMB(0x80072a66d512100), CNST_LIMB(0xffe357ff59e6a004)},
                /* 167 */ {8, CNST_LIMB(0x22abc300df54ca7c), CNST_LIMB(0xec474e39705912d2), CNST_LIMB(0x86546633b42b9c1), CNST_LIMB(0xe7dfd1be05fa61a8)},
                /* 168 */ {8, CNST_LIMB(0x22a16b90698da5d2), CNST_LIMB(0xec8ddd448f8b845a), CNST_LIMB(0x8ce6b0861000000), CNST_LIMB(0xd11ed6fc78f760e5)},
                /* 169 */ {8, CNST_LIMB(0x229729f1b2c83ded), CNST_LIMB(0xecd4011c8f11979a), CNST_LIMB(0x93c08e16a022441), CNST_LIMB(0xbb8db609dd29ebfe)},
                /* 170 */ {8, CNST_LIMB(0x228cfdd444992f78), CNST_LIMB(0xed19bb053fb0284e), CNST_LIMB(0x9ae49717f026100), CNST_LIMB(0xa71aec8d1813d532)},
                /* 171 */ {8, CNST_LIMB(0x2282e6e94ccb8588), CNST_LIMB(0xed5f0c3cbf8fa470), CNST_LIMB(0xa25577ae24c1a61), CNST_LIMB(0x93b612a9f20fbc02)},
                /* 172 */ {8, CNST_LIMB(0x2278e4e392557ecf), CNST_LIMB(0xeda3f5fb9c415052), CNST_LIMB(0xaa15f068e610000), CNST_LIMB(0x814fc7b19a67d317)},
                /* 173 */ {8, CNST_LIMB(0x226ef7776aa7fd29), CNST_LIMB(0xede87974f3c81855), CNST_LIMB(0xb228d6bf7577921), CNST_LIMB(0x6fd9a03f2e0a4b7c)},
                /* 174 */ {8, CNST_LIMB(0x22651e5aaf5532d0), CNST_LIMB(0xee2c97d694adab3f), CNST_LIMB(0xba91158ef5c4100), CNST_LIMB(0x5f4615a38d0d316e)},
                /* 175 */ {8, CNST_LIMB(0x225b5944b40b4694), CNST_LIMB(0xee7052491d2c3e64), CNST_LIMB(0xc351ad9aec0b681), CNST_LIMB(0x4f8876863479a286)},
                /* 176 */ {8, CNST_LIMB(0x2251a7ee3cdfcca5), CNST_LIMB(0xeeb3a9f01975077f), CNST_LIMB(0xcc6db6100000000), CNST_LIMB(0x4094d8a3041b60eb)},
                /* 177 */ {8, CNST_LIMB(0x22480a1174e913d9), CNST_LIMB(0xeef69fea211b2627), CNST_LIMB(0xd5e85d09025c181), CNST_LIMB(0x32600b8ed883a09b)},
                /* 178 */ {8, CNST_LIMB(0x223e7f69e522683c), CNST_LIMB(0xef393550f3aa6906), CNST_LIMB(0xdfc4e816401c100), CNST_LIMB(0x24df8c6eb4b6d1f1)},
                /* 179 */ {8, CNST_LIMB(0x223507b46b988abe), CNST_LIMB(0xef7b6b399471103e), CNST_LIMB(0xea06b4c72947221), CNST_LIMB(0x18097a8ee151acef)},
                /* 180 */ {8, CNST_LIMB(0x222ba2af32dbbb9e), CNST_LIMB(0xefbd42b465836767), CNST_LIMB(0xf4b139365210000), CNST_LIMB(0xbd48cc8ec1cd8e3)},
                /* 181 */ {8, CNST_LIMB(0x22225019a9b4d16c), CNST_LIMB(0xeffebccd41ffcd5c), CNST_LIMB(0xffc80497d520961), CNST_LIMB(0x3807a8d67485fb)},
                /* 182 */ {8, CNST_LIMB(0x22190fb47b1af172), CNST_LIMB(0xf03fda8b97997f33), CNST_LIMB(0x10b4ebfca1dee100), CNST_LIMB(0xea5768860b62e8d8)},
                /* 183 */ {8, CNST_LIMB(0x220fe14186679801), CNST_LIMB(0xf0809cf27f703d52), CNST_LIMB(0x117492de921fc141), CNST_LIMB(0xd54faf5b635c5005)},
                /* 184 */ {8, CNST_LIMB(0x2206c483d7c6b786), CNST_LIMB(0xf0c10500d63aa658), CNST_LIMB(0x123bb2ce41000000), CNST_LIMB(0xc14a56233a377926)},
                /* 185 */ {8, CNST_LIMB(0x21fdb93fa0e0ccc5), CNST_LIMB(0xf10113b153c8ea7b), CNST_LIMB(0x130a8b6157bdecc1), CNST_LIMB(0xae39a88db7cd329f)},
                /* 186 */ {8, CNST_LIMB(0x21f4bf3a31bcdcaa), CNST_LIMB(0xf140c9faa1e5439e), CNST_LIMB(0x13e15dede0e8a100), CNST_LIMB(0x9c10bde69efa7ab6)},
                /* 187 */ {8, CNST_LIMB(0x21ebd639f1d86584), CNST_LIMB(0xf18028cf72976a4e), CNST_LIMB(0x14c06d941c0ca7e1), CNST_LIMB(0x8ac36c42a2836497)},
                /* 188 */ {8, CNST_LIMB(0x21e2fe06597361a6), CNST_LIMB(0xf1bf311e95d00de3), CNST_LIMB(0x15a7ff487a810000), CNST_LIMB(0x7a463c8b84f5ef67)},
                /* 189 */ {8, CNST_LIMB(0x21da3667eb0e8ccb), CNST_LIMB(0xf1fde3d30e812642), CNST_LIMB(0x169859ddc5c697a1), CNST_LIMB(0x6a8e5f5ad090fd4b)},
                /* 190 */ {8, CNST_LIMB(0x21d17f282d1a300e), CNST_LIMB(0xf23c41d42727c808), CNST_LIMB(0x1791c60f6fed0100), CNST_LIMB(0x5b91a2943596fc56)},
                /* 191 */ {8, CNST_LIMB(0x21c8d811a3d3c9e1), CNST_LIMB(0xf27a4c0585cbf805), CNST_LIMB(0x18948e8c0e6fba01), CNST_LIMB(0x4d4667b1c468e8f0)},
                /* 192 */ {8, CNST_LIMB(0x21c040efcb50f858), CNST_LIMB(0xf2b803473f7ad0f3), CNST_LIMB(0x19a1000000000000), CNST_LIMB(0x3fa39ab547994daf)},
                /* 193 */ {8, CNST_LIMB(0x21b7b98f11b61c1a), CNST_LIMB(0xf2f56875eb3f2614), CNST_LIMB(0x1ab769203dafc601), CNST_LIMB(0x32a0a9b2faee1e2a)},
                /* 194 */ {8, CNST_LIMB(0x21af41bcd19739ba), CNST_LIMB(0xf3327c6ab49ca6c8), CNST_LIMB(0x1bd81ab557f30100), CNST_LIMB(0x26357ceac0e96962)},
                /* 195 */ {8, CNST_LIMB(0x21a6d9474c81adf0), CNST_LIMB(0xf36f3ffb6d916240), CNST_LIMB(0x1d0367a69fed1ba1), CNST_LIMB(0x1a5a6f65caa5859e)},
                /* 196 */ {8, CNST_LIMB(0x219e7ffda5ad572a), CNST_LIMB(0xf3abb3faa02166cc), CNST_LIMB(0x1e39a5057d810000), CNST_LIMB(0xf08480f672b4e86)},
                /* 197 */ {8, CNST_LIMB(0x219635afdcd3e46d), CNST_LIMB(0xf3e7d9379f70166a), CNST_LIMB(0x1f7b2a18f29ac3e1), CNST_LIMB(0x4383340615612ca)},
                /* 198 */ {8, CNST_LIMB(0x218dfa2ec92d0643), CNST_LIMB(0xf423b07e986aa967), CNST_LIMB(0x20c850694c2aa100), CNST_LIMB(0xf3c77969ee4be5a2)},
                /* 199 */ {8, CNST_LIMB(0x2185cd4c148e4ae2), CNST_LIMB(0xf45f3a98a20738a4), CNST_LIMB(0x222173cc014980c1), CNST_LIMB(0xe00993cc187c5ec9)},
                /* 200 */ {8, CNST_LIMB(0x217daeda36ad7a5c), CNST_LIMB(0xf49a784bcd1b8afe), CNST_LIMB(0x2386f26fc1000000), CNST_LIMB(0xcd2b297d889bc2b6)},
                /* 201 */ {8, CNST_LIMB(0x21759eac708452fe), CNST_LIMB(0xf4d56a5b33cec44a), CNST_LIMB(0x24f92ce8af296d41), CNST_LIMB(0xbb214d5064862b22)},
                /* 202 */ {8, CNST_LIMB(0x216d9c96c7d490d4), CNST_LIMB(0xf510118708a8f8dd), CNST_LIMB(0x2678863cd0ece100), CNST_LIMB(0xa9e1a7ca7ea10e20)},
                /* 203 */ {8, CNST_LIMB(0x2165a86e02cb358c), CNST_LIMB(0xf54a6e8ca5438db1), CNST_LIMB(0x280563f0a9472d61), CNST_LIMB(0x99626e72b39ea0cf)},
                /* 204 */ {8, CNST_LIMB(0x215dc207a3c20fdf), CNST_LIMB(0xf5848226989d33c3), CNST_LIMB(0x29a02e1406210000), CNST_LIMB(0x899a5ba9c13fafd9)},
                /* 205 */ {8, CNST_LIMB(0x2155e939e51e8b37), CNST_LIMB(0xf5be4d0cb51434aa), CNST_LIMB(0x2b494f4efe6d2e21), CNST_LIMB(0x7a80a705391e96ff)},
                /* 206 */ {8, CNST_LIMB(0x214e1ddbb54cd933), CNST_LIMB(0xf5f7cff41e09aeb8), CNST_LIMB(0x2d0134ef21cbc100), CNST_LIMB(0x6c0cfe23de23042a)},
                /* 207 */ {8, CNST_LIMB(0x21465fc4b2d68f98), CNST_LIMB(0xf6310b8f55304840), CNST_LIMB(0x2ec84ef4da2ef581), CNST_LIMB(0x5e377df359c944dd)},
                /* 208 */ {8, CNST_LIMB(0x213eaecd2893dd60), CNST_LIMB(0xf66a008e4788cbcd), CNST_LIMB(0x309f102100000000), CNST_LIMB(0x50f8ac5fc8f53985)},
                /* 209 */ {8, CNST_LIMB(0x21370ace09f681c6), CNST_LIMB(0xf6a2af9e5a0f0a08), CNST_LIMB(0x3285ee02a1420281), CNST_LIMB(0x44497266278e35b7)},
                /* 210 */ {8, CNST_LIMB(0x212f73a0ef6db7cb), CNST_LIMB(0xf6db196a761949d9), CNST_LIMB(0x347d6104fc324100), CNST_LIMB(0x382316831f7ee175)},
                /* 211 */ {8, CNST_LIMB(0x2127e92012e25004), CNST_LIMB(0xf7133e9b156c7be5), CNST_LIMB(0x3685e47dade53d21), CNST_LIMB(0x2c7f377833b8946e)},
                /* 212 */ {8, CNST_LIMB(0x21206b264c4a39a7), CNST_LIMB(0xf74b1fd64e0753c6), CNST_LIMB(0x389ff6bb15610000), CNST_LIMB(0x2157c761ab4163ef)},
                /* 213 */ {8, CNST_LIMB(0x2118f98f0e52c28f), CNST_LIMB(0xf782bdbfdda6577b), CNST_LIMB(0x3acc1912ebb57661), CNST_LIMB(0x16a7071803cc49a9)},
                /* 214 */ {8, CNST_LIMB(0x211194366320dc66), CNST_LIMB(0xf7ba18f93502e409), CNST_LIMB(0x3d0acff111946100), CNST_LIMB(0xc6781d80f8224fc)},
                /* 215 */ {8, CNST_LIMB(0x210a3af8e926bb78), CNST_LIMB(0xf7f1322182cf15d1), CNST_LIMB(0x3f5ca2e692eaf841), CNST_LIMB(0x294092d370a900b)},
                /* 216 */ {8, CNST_LIMB(0x2102edb3d00e29a6), CNST_LIMB(0xf82809d5be7072db), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295)},
                /* 217 */ {8, CNST_LIMB(0x20fbac44d5b6edc2), CNST_LIMB(0xf85ea0b0b27b2610), CNST_LIMB(0x443bcb714399a5c1), CNST_LIMB(0xe03b98f103fad6d2)},
                /* 218 */ {8, CNST_LIMB(0x20f4768a4348ad08), CNST_LIMB(0xf894f74b06ef8b40), CNST_LIMB(0x46ca406c81af2100), CNST_LIMB(0xcee3d32cad2a9049)},
                /* 219 */ {8, CNST_LIMB(0x20ed4c62ea57b1f0), CNST_LIMB(0xf8cb0e3b4b3bbdb3), CNST_LIMB(0x496e106ac22aaae1), CNST_LIMB(0xbe3f9df9277fdada)},
                /* 220 */ {8, CNST_LIMB(0x20e62dae221c087a), CNST_LIMB(0xf900e6160002ccfe), CNST_LIMB(0x4c27d39fa5410000), CNST_LIMB(0xae46f0d94c05e933)},
                /* 221 */ {8, CNST_LIMB(0x20df1a4bc4ba6525), CNST_LIMB(0xf9367f6da0ab2e9c), CNST_LIMB(0x4ef825c296e43ca1), CNST_LIMB(0x9ef2280fb437a33d)},
                /* 222 */ {8, CNST_LIMB(0x20d8121c2c9e506e), CNST_LIMB(0xf96bdad2acb5f5ef), CNST_LIMB(0x51dfa61f5ad88100), CNST_LIMB(0x9039ff426d3f284b)},
                /* 223 */ {8, CNST_LIMB(0x20d1150031e51549), CNST_LIMB(0xf9a0f8d3b0e04fde), CNST_LIMB(0x54def7a6d2f16901), CNST_LIMB(0x82178c6d6b51f8f4)},
                /* 224 */ {8, CNST_LIMB(0x20ca22d927d8f54d), CNST_LIMB(0xf9d5d9fd5010b366), CNST_LIMB(0x57f6c10000000000), CNST_LIMB(0x74843b1ee4c1e053)},
                /* 225 */ {8, CNST_LIMB(0x20c33b88da7c29aa), CNST_LIMB(0xfa0a7eda4c112ce6), CNST_LIMB(0x5b27ac993df97701), CNST_LIMB(0x6779c7f90dc42f48)},
                /* 226 */ {8, CNST_LIMB(0x20bc5ef18c233bdf), CNST_LIMB(0xfa3ee7f38e181ed0), CNST_LIMB(0x5e7268b9bbdf8100), CNST_LIMB(0x5af23c74f9ad9fe9)},
                /* 227 */ {8, CNST_LIMB(0x20b58cf5f31e4526), CNST_LIMB(0xfa7315d02f20c7bd), CNST_LIMB(0x61d7a7932ff3d6a1), CNST_LIMB(0x4ee7eae2acdc617e)},
                /* 228 */ {8, CNST_LIMB(0x20aec5793770a74d), CNST_LIMB(0xfaa708f58014d37c), CNST_LIMB(0x65581f53c8c10000), CNST_LIMB(0x43556aa2ac262a0b)},
                /* 229 */ {8, CNST_LIMB(0x20a8085ef096d530), CNST_LIMB(0xfadac1e711c832d1), CNST_LIMB(0x68f48a385b8320e1), CNST_LIMB(0x3835949593b8ddd1)},
                /* 230 */ {8, CNST_LIMB(0x20a1558b2359c4b1), CNST_LIMB(0xfb0e4126bcc86bd7), CNST_LIMB(0x6cada69ed07c2100), CNST_LIMB(0x2d837fbe78458762)},
                /* 231 */ {8, CNST_LIMB(0x209aace23fafa72e), CNST_LIMB(0xfb418734a9008bd9), CNST_LIMB(0x70843718cdbf27c1), CNST_LIMB(0x233a7e150a54a555)},
                /* 232 */ {8, CNST_LIMB(0x20940e491ea988d7), CNST_LIMB(0xfb74948f5532da4b), CNST_LIMB(0x7479027ea1000000), CNST_LIMB(0x19561984a50ff8fe)},
                /* 233 */ {8, CNST_LIMB(0x208d79a5006d7a47), CNST_LIMB(0xfba769b39e49640e), CNST_LIMB(0x788cd40268f39641), CNST_LIMB(0xfd211159fe3490f)},
                /* 234 */ {8, CNST_LIMB(0x2086eedb8a3cead3), CNST_LIMB(0xfbda071cc67e6db5), CNST_LIMB(0x7cc07b437ecf6100), CNST_LIMB(0x6aa563e655033e3)},
                /* 235 */ {8, CNST_LIMB(0x20806dd2c486dcc6), CNST_LIMB(0xfc0c6d447c5dd362), CNST_LIMB(0x8114cc6220762061), CNST_LIMB(0xfbb614b3f2d3b14c)},
                /* 236 */ {8, CNST_LIMB(0x2079f67119059fae), CNST_LIMB(0xfc3e9ca2e1a05533), CNST_LIMB(0x858aa0135be10000), CNST_LIMB(0xeac0f8837fb05773)},
                /* 237 */ {8, CNST_LIMB(0x2073889d50e7bf63), CNST_LIMB(0xfc7095ae91e1c760), CNST_LIMB(0x8a22d3b53c54c321), CNST_LIMB(0xda6e4c10e8615ca5)},
                /* 238 */ {8, CNST_LIMB(0x206d243e9303d929), CNST_LIMB(0xfca258dca9331635), CNST_LIMB(0x8ede496339f34100), CNST_LIMB(0xcab755a8d01fa67f)},
                /* 239 */ {8, CNST_LIMB(0x2066c93c62170aa8), CNST_LIMB(0xfcd3e6a0ca8906c2), CNST_LIMB(0x93bde80aec3a1481), CNST_LIMB(0xbb95a9ae71aa3e0c)},
                /* 240 */ {8, CNST_LIMB(0x2060777e9b0db0f6), CNST_LIMB(0xfd053f6d26089673), CNST_LIMB(0x98c29b8100000000), CNST_LIMB(0xad0326c296b4f529)},
                /* 241 */ {8, CNST_LIMB(0x205a2eed73563032), CNST_LIMB(0xfd3663b27f31d529), CNST_LIMB(0x9ded549671832381), CNST_LIMB(0x9ef9f21eed31b7c1)},
                /* 242 */ {8, CNST_LIMB(0x2053ef71773d7e6a), CNST_LIMB(0xfd6753e032ea0efe), CNST_LIMB(0xa33f092e0b1ac100), CNST_LIMB(0x91747422be14b0b2)},
                /* 243 */ {8, CNST_LIMB(0x204db8f388552ea9), CNST_LIMB(0xfd9810643d6614c3), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
                /* 244 */ {8, CNST_LIMB(0x20478b5cdbe2bb2f), CNST_LIMB(0xfdc899ab3ff56c5e), CNST_LIMB(0xae5b564ac3a10000), CNST_LIMB(0x77df79e9a96c06f6)},
                /* 245 */ {8, CNST_LIMB(0x20416696f957cfbf), CNST_LIMB(0xfdf8f02086af2c4b), CNST_LIMB(0xb427f4b3be74c361), CNST_LIMB(0x6bc6019636c7d0c2)},
                /* 246 */ {8, CNST_LIMB(0x203b4a8bb8d356e7), CNST_LIMB(0xfe29142e0e01401f), CNST_LIMB(0xba1f9a938041e100), CNST_LIMB(0x601c4205aebd9e47)},
                /* 247 */ {8, CNST_LIMB(0x2035372541ab0f0d), CNST_LIMB(0xfe59063c8822ce56), CNST_LIMB(0xc0435871d1110f41), CNST_LIMB(0x54ddc59756f05016)},
                /* 248 */ {8, CNST_LIMB(0x202f2c4e08fd6dcc), CNST_LIMB(0xfe88c6b3626a72aa), CNST_LIMB(0xc694446f01000000), CNST_LIMB(0x4a0648979c838c18)},
                /* 249 */ {8, CNST_LIMB(0x202929f0d04b99e9), CNST_LIMB(0xfeb855f8ca88fb0d), CNST_LIMB(0xcd137a5b57ac3ec1), CNST_LIMB(0x3f91b6e0bb3a053d)},
                /* 250 */ {8, CNST_LIMB(0x20232ff8a41b45eb), CNST_LIMB(0xfee7b471b3a9507d), CNST_LIMB(0xd3c21bcecceda100), CNST_LIMB(0x357c299a88ea76a5)},
                /* 251 */ {8, CNST_LIMB(0x201d3e50daa036db), CNST_LIMB(0xff16e281db76303b), CNST_LIMB(0xdaa150410b788de1), CNST_LIMB(0x2bc1e517aecc56e3)},
                /* 252 */ {8, CNST_LIMB(0x201754e5126d446d), CNST_LIMB(0xff45e08bcf06554e), CNST_LIMB(0xe1b24521be010000), CNST_LIMB(0x225f56ceb3da9f5d)},
                /* 253 */ {8, CNST_LIMB(0x201173a1312ca135), CNST_LIMB(0xff74aef0efafadd7), CNST_LIMB(0xe8f62df12777c1a1), CNST_LIMB(0x1951136d53ad63ac)},
                /* 254 */ {8, CNST_LIMB(0x200b9a71625f3b13), CNST_LIMB(0xffa34e1177c23362), CNST_LIMB(0xf06e445906fc0100), CNST_LIMB(0x1093d504b3cd7d93)},
                /* 255 */ {8, CNST_LIMB(0x2005c94216230568), CNST_LIMB(0xffd1be4c7f2af942), CNST_LIMB(0xf81bc845c81bf801), CNST_LIMB(0x824794d1ec1814f)},
                /* 256 */ {8, CNST_LIMB(0x1fffffffffffffff), CNST_LIMB(0xffffffffffffffff), CNST_LIMB(0x8), CNST_LIMB(0x0)},
        };

        __device__ const mp_limb_t
            __gmp_fib_table[FIB_TABLE_LIMIT + 2] = {
                CNST_LIMB(0x1),                /* -1 */
                CNST_LIMB(0x0),                /* 0 */
                CNST_LIMB(0x1),                /* 1 */
                CNST_LIMB(0x1),                /* 2 */
                CNST_LIMB(0x2),                /* 3 */
                CNST_LIMB(0x3),                /* 4 */
                CNST_LIMB(0x5),                /* 5 */
                CNST_LIMB(0x8),                /* 6 */
                CNST_LIMB(0xd),                /* 7 */
                CNST_LIMB(0x15),               /* 8 */
                CNST_LIMB(0x22),               /* 9 */
                CNST_LIMB(0x37),               /* 10 */
                CNST_LIMB(0x59),               /* 11 */
                CNST_LIMB(0x90),               /* 12 */
                CNST_LIMB(0xe9),               /* 13 */
                CNST_LIMB(0x179),              /* 14 */
                CNST_LIMB(0x262),              /* 15 */
                CNST_LIMB(0x3db),              /* 16 */
                CNST_LIMB(0x63d),              /* 17 */
                CNST_LIMB(0xa18),              /* 18 */
                CNST_LIMB(0x1055),             /* 19 */
                CNST_LIMB(0x1a6d),             /* 20 */
                CNST_LIMB(0x2ac2),             /* 21 */
                CNST_LIMB(0x452f),             /* 22 */
                CNST_LIMB(0x6ff1),             /* 23 */
                CNST_LIMB(0xb520),             /* 24 */
                CNST_LIMB(0x12511),            /* 25 */
                CNST_LIMB(0x1da31),            /* 26 */
                CNST_LIMB(0x2ff42),            /* 27 */
                CNST_LIMB(0x4d973),            /* 28 */
                CNST_LIMB(0x7d8b5),            /* 29 */
                CNST_LIMB(0xcb228),            /* 30 */
                CNST_LIMB(0x148add),           /* 31 */
                CNST_LIMB(0x213d05),           /* 32 */
                CNST_LIMB(0x35c7e2),           /* 33 */
                CNST_LIMB(0x5704e7),           /* 34 */
                CNST_LIMB(0x8cccc9),           /* 35 */
                CNST_LIMB(0xe3d1b0),           /* 36 */
                CNST_LIMB(0x1709e79),          /* 37 */
                CNST_LIMB(0x2547029),          /* 38 */
                CNST_LIMB(0x3c50ea2),          /* 39 */
                CNST_LIMB(0x6197ecb),          /* 40 */
                CNST_LIMB(0x9de8d6d),          /* 41 */
                CNST_LIMB(0xff80c38),          /* 42 */
                CNST_LIMB(0x19d699a5),         /* 43 */
                CNST_LIMB(0x29cea5dd),         /* 44 */
                CNST_LIMB(0x43a53f82),         /* 45 */
                CNST_LIMB(0x6d73e55f),         /* 46 */
                CNST_LIMB(0xb11924e1),         /* 47 */
                CNST_LIMB(0x11e8d0a40),        /* 48 */
                CNST_LIMB(0x1cfa62f21),        /* 49 */
                CNST_LIMB(0x2ee333961),        /* 50 */
                CNST_LIMB(0x4bdd96882),        /* 51 */
                CNST_LIMB(0x7ac0ca1e3),        /* 52 */
                CNST_LIMB(0xc69e60a65),        /* 53 */
                CNST_LIMB(0x1415f2ac48),       /* 54 */
                CNST_LIMB(0x207fd8b6ad),       /* 55 */
                CNST_LIMB(0x3495cb62f5),       /* 56 */
                CNST_LIMB(0x5515a419a2),       /* 57 */
                CNST_LIMB(0x89ab6f7c97),       /* 58 */
                CNST_LIMB(0xdec1139639),       /* 59 */
                CNST_LIMB(0x1686c8312d0),      /* 60 */
                CNST_LIMB(0x2472d96a909),      /* 61 */
                CNST_LIMB(0x3af9a19bbd9),      /* 62 */
                CNST_LIMB(0x5f6c7b064e2),      /* 63 */
                CNST_LIMB(0x9a661ca20bb),      /* 64 */
                CNST_LIMB(0xf9d297a859d),      /* 65 */
                CNST_LIMB(0x19438b44a658),     /* 66 */
                CNST_LIMB(0x28e0b4bf2bf5),     /* 67 */
                CNST_LIMB(0x42244003d24d),     /* 68 */
                CNST_LIMB(0x6b04f4c2fe42),     /* 69 */
                CNST_LIMB(0xad2934c6d08f),     /* 70 */
                CNST_LIMB(0x1182e2989ced1),    /* 71 */
                CNST_LIMB(0x1c5575e509f60),    /* 72 */
                CNST_LIMB(0x2dd8587da6e31),    /* 73 */
                CNST_LIMB(0x4a2dce62b0d91),    /* 74 */
                CNST_LIMB(0x780626e057bc2),    /* 75 */
                CNST_LIMB(0xc233f54308953),    /* 76 */
                CNST_LIMB(0x13a3a1c2360515),   /* 77 */
                CNST_LIMB(0x1fc6e116668e68),   /* 78 */
                CNST_LIMB(0x336a82d89c937d),   /* 79 */
                CNST_LIMB(0x533163ef0321e5),   /* 80 */
                CNST_LIMB(0x869be6c79fb562),   /* 81 */
                CNST_LIMB(0xd9cd4ab6a2d747),   /* 82 */
                CNST_LIMB(0x16069317e428ca9),  /* 83 */
                CNST_LIMB(0x23a367c34e563f0),  /* 84 */
                CNST_LIMB(0x39a9fadb327f099),  /* 85 */
                CNST_LIMB(0x5d4d629e80d5489),  /* 86 */
                CNST_LIMB(0x96f75d79b354522),  /* 87 */
                CNST_LIMB(0xf444c01834299ab),  /* 88 */
                CNST_LIMB(0x18b3c1d91e77decd), /* 89 */
                CNST_LIMB(0x27f80ddaa1ba7878), /* 90 */
                CNST_LIMB(0x40abcfb3c0325745), /* 91 */
                CNST_LIMB(0x68a3dd8e61eccfbd), /* 92 */
                CNST_LIMB(0xa94fad42221f2702), /* 93 */
        };

#else
__GPGMP_DECLSPEC const struct bases mp_bases[257] =
    {
        /*   0 */ {0, 0, 0, 0, 0},
        /*   1 */ {0, 0, 0, 0, 0},
        /*   2 */ {64, CNST_LIMB(0xffffffffffffffff), CNST_LIMB(0x1fffffffffffffff), CNST_LIMB(0x1), CNST_LIMB(0x0)},
        /*   3 */ {40, CNST_LIMB(0xa1849cc1a9a9e94e), CNST_LIMB(0x32b803473f7ad0f3), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
        /*   4 */ {32, CNST_LIMB(0x7fffffffffffffff), CNST_LIMB(0x3fffffffffffffff), CNST_LIMB(0x2), CNST_LIMB(0x0)},
        /*   5 */ {27, CNST_LIMB(0x6e40d1a4143dcb94), CNST_LIMB(0x4a4d3c25e68dc57f), CNST_LIMB(0x6765c793fa10079d), CNST_LIMB(0x3ce9a36f23c0fc90)},
        /*   6 */ {24, CNST_LIMB(0x6308c91b702a7cf4), CNST_LIMB(0x52b803473f7ad0f3), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295)},
        /*   7 */ {22, CNST_LIMB(0x5b3064eb3aa6d388), CNST_LIMB(0x59d5d9fd5010b366), CNST_LIMB(0x3642798750226111), CNST_LIMB(0x2df495ccaa57147b)},
        /*   8 */ {21, CNST_LIMB(0x5555555555555555), CNST_LIMB(0x5fffffffffffffff), CNST_LIMB(0x3), CNST_LIMB(0x0)},
        /*   9 */ {20, CNST_LIMB(0x50c24e60d4d4f4a7), CNST_LIMB(0x6570068e7ef5a1e7), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
        /*  10 */ {19, CNST_LIMB(0x4d104d427de7fbcc), CNST_LIMB(0x6a4d3c25e68dc57f), CNST_LIMB(0x8ac7230489e80000), CNST_LIMB(0xd83c94fb6d2ac34a)},
        /*  11 */ {18, CNST_LIMB(0x4a00270775914e88), CNST_LIMB(0x6eb3a9f01975077f), CNST_LIMB(0x4d28cb56c33fa539), CNST_LIMB(0xa8adf7ae45e7577b)},
        /*  12 */ {17, CNST_LIMB(0x4768ce0d05818e12), CNST_LIMB(0x72b803473f7ad0f3), CNST_LIMB(0x1eca170c00000000), CNST_LIMB(0xa10c2bec5da8f8f)},
        /*  13 */ {17, CNST_LIMB(0x452e53e365907bda), CNST_LIMB(0x766a008e4788cbcd), CNST_LIMB(0x780c7372621bd74d), CNST_LIMB(0x10f4becafe412ec3)},
        /*  14 */ {16, CNST_LIMB(0x433cfffb4b5aae55), CNST_LIMB(0x79d5d9fd5010b366), CNST_LIMB(0x1e39a5057d810000), CNST_LIMB(0xf08480f672b4e86)},
        /*  15 */ {16, CNST_LIMB(0x41867711b4f85355), CNST_LIMB(0x7d053f6d26089673), CNST_LIMB(0x5b27ac993df97701), CNST_LIMB(0x6779c7f90dc42f48)},
        /*  16 */ {16, CNST_LIMB(0x3fffffffffffffff), CNST_LIMB(0x7fffffffffffffff), CNST_LIMB(0x4), CNST_LIMB(0x0)},
        /*  17 */ {15, CNST_LIMB(0x3ea16afd58b10966), CNST_LIMB(0x82cc7edf592262cf), CNST_LIMB(0x27b95e997e21d9f1), CNST_LIMB(0x9c71e11bab279323)},
        /*  18 */ {15, CNST_LIMB(0x3d64598d154dc4de), CNST_LIMB(0x8570068e7ef5a1e7), CNST_LIMB(0x5da0e1e53c5c8000), CNST_LIMB(0x5dfaa697ec6f6a1c)},
        /*  19 */ {15, CNST_LIMB(0x3c43c23018bb5563), CNST_LIMB(0x87ef05ae409a0288), CNST_LIMB(0xd2ae3299c1c4aedb), CNST_LIMB(0x3711783f6be7e9ec)},
        /*  20 */ {14, CNST_LIMB(0x3b3b9a42873069c7), CNST_LIMB(0x8a4d3c25e68dc57f), CNST_LIMB(0x16bcc41e90000000), CNST_LIMB(0x6849b86a12b9b01e)},
        /*  21 */ {14, CNST_LIMB(0x3a4898f06cf41ac9), CNST_LIMB(0x8c8ddd448f8b845a), CNST_LIMB(0x2d04b7fdd9c0ef49), CNST_LIMB(0x6bf097ba5ca5e239)},
        /*  22 */ {14, CNST_LIMB(0x39680b13582e7c18), CNST_LIMB(0x8eb3a9f01975077f), CNST_LIMB(0x5658597bcaa24000), CNST_LIMB(0x7b8015c8d7af8f08)},
        /*  23 */ {14, CNST_LIMB(0x3897b2b751ae561a), CNST_LIMB(0x90c10500d63aa658), CNST_LIMB(0xa0e2073737609371), CNST_LIMB(0x975a24b3a3151b38)},
        /*  24 */ {13, CNST_LIMB(0x37d5aed131f19c98), CNST_LIMB(0x92b803473f7ad0f3), CNST_LIMB(0xc29e98000000000), CNST_LIMB(0x50bd367972689db1)},
        /*  25 */ {13, CNST_LIMB(0x372068d20a1ee5ca), CNST_LIMB(0x949a784bcd1b8afe), CNST_LIMB(0x14adf4b7320334b9), CNST_LIMB(0x8c240c4aecb13bb5)},
        /*  26 */ {13, CNST_LIMB(0x3676867e5d60de29), CNST_LIMB(0x966a008e4788cbcd), CNST_LIMB(0x226ed36478bfa000), CNST_LIMB(0xdbd2e56854e118c9)},
        /*  27 */ {13, CNST_LIMB(0x35d6deeb388df86f), CNST_LIMB(0x982809d5be7072db), CNST_LIMB(0x383d9170b85ff80b), CNST_LIMB(0x2351ffcaa9c7c4ae)},
        /*  28 */ {13, CNST_LIMB(0x354071d61c77fa2e), CNST_LIMB(0x99d5d9fd5010b366), CNST_LIMB(0x5a3c23e39c000000), CNST_LIMB(0x6b24188ca33b0636)},
        /*  29 */ {13, CNST_LIMB(0x34b260c5671b18ac), CNST_LIMB(0x9b74948f5532da4b), CNST_LIMB(0x8e65137388122bcd), CNST_LIMB(0xcc3dceaf2b8ba99d)},
        /*  30 */ {13, CNST_LIMB(0x342be986572b45cc), CNST_LIMB(0x9d053f6d26089673), CNST_LIMB(0xdd41bb36d259e000), CNST_LIMB(0x2832e835c6c7d6b6)},
        /*  31 */ {12, CNST_LIMB(0x33ac61b998fbbdf2), CNST_LIMB(0x9e88c6b3626a72aa), CNST_LIMB(0xaee5720ee830681), CNST_LIMB(0x76b6aa272e1873c5)},
        /*  32 */ {12, CNST_LIMB(0x3333333333333333), CNST_LIMB(0x9fffffffffffffff), CNST_LIMB(0x5), CNST_LIMB(0x0)},
        /*  33 */ {12, CNST_LIMB(0x32bfd90114c12861), CNST_LIMB(0xa16bad3758efd873), CNST_LIMB(0x172588ad4f5f0981), CNST_LIMB(0x61eaf5d402c7bf4f)},
        /*  34 */ {12, CNST_LIMB(0x3251dcf6169e45f2), CNST_LIMB(0xa2cc7edf592262cf), CNST_LIMB(0x211e44f7d02c1000), CNST_LIMB(0xeeb658123ffb27ec)},
        /*  35 */ {12, CNST_LIMB(0x31e8d59f180dc630), CNST_LIMB(0xa4231623369e78e5), CNST_LIMB(0x2ee56725f06e5c71), CNST_LIMB(0x5d5e3762e6fdf509)},
        /*  36 */ {12, CNST_LIMB(0x3184648db8153e7a), CNST_LIMB(0xa570068e7ef5a1e7), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295)},
        /*  37 */ {12, CNST_LIMB(0x312434e89c35dacd), CNST_LIMB(0xa6b3d78b6d3b24fb), CNST_LIMB(0x5b5b57f8a98a5dd1), CNST_LIMB(0x66ae7831762efb6f)},
        /*  38 */ {12, CNST_LIMB(0x30c7fa349460a541), CNST_LIMB(0xa7ef05ae409a0288), CNST_LIMB(0x7dcff8986ea31000), CNST_LIMB(0x47388865a00f544)},
        /*  39 */ {12, CNST_LIMB(0x306f6f4c8432bc6d), CNST_LIMB(0xa92203d587039cc1), CNST_LIMB(0xabd4211662a6b2a1), CNST_LIMB(0x7d673c33a123b54c)},
        /*  40 */ {12, CNST_LIMB(0x301a557ffbfdd252), CNST_LIMB(0xaa4d3c25e68dc57f), CNST_LIMB(0xe8d4a51000000000), CNST_LIMB(0x19799812dea11197)},
        /*  41 */ {11, CNST_LIMB(0x2fc873d1fda55f3b), CNST_LIMB(0xab7110e6ce866f2b), CNST_LIMB(0x7a32956ad081b79), CNST_LIMB(0xc27e62e0686feae)},
        /*  42 */ {11, CNST_LIMB(0x2f799652a4e6dc49), CNST_LIMB(0xac8ddd448f8b845a), CNST_LIMB(0x9f49aaff0e86800), CNST_LIMB(0x9b6e7507064ce7c7)},
        /*  43 */ {11, CNST_LIMB(0x2f2d8d8f64460aad), CNST_LIMB(0xada3f5fb9c415052), CNST_LIMB(0xce583bb812d37b3), CNST_LIMB(0x3d9ac2bf66cfed94)},
        /*  44 */ {11, CNST_LIMB(0x2ee42e164e8f53a4), CNST_LIMB(0xaeb3a9f01975077f), CNST_LIMB(0x109b79a654c00000), CNST_LIMB(0xed46bc50ce59712a)},
        /*  45 */ {11, CNST_LIMB(0x2e9d500984041dbd), CNST_LIMB(0xafbd42b465836767), CNST_LIMB(0x1543beff214c8b95), CNST_LIMB(0x813d97e2c89b8d46)},
        /*  46 */ {11, CNST_LIMB(0x2e58cec05a6a8144), CNST_LIMB(0xb0c10500d63aa658), CNST_LIMB(0x1b149a79459a3800), CNST_LIMB(0x2e81751956af8083)},
        /*  47 */ {11, CNST_LIMB(0x2e1688743ef9104c), CNST_LIMB(0xb1bf311e95d00de3), CNST_LIMB(0x224edfb5434a830f), CNST_LIMB(0xdd8e0a95e30c0988)},
        /*  48 */ {11, CNST_LIMB(0x2dd65df7a583598f), CNST_LIMB(0xb2b803473f7ad0f3), CNST_LIMB(0x2b3fb00000000000), CNST_LIMB(0x7ad4dd48a0b5b167)},
        /*  49 */ {11, CNST_LIMB(0x2d9832759d5369c4), CNST_LIMB(0xb3abb3faa02166cc), CNST_LIMB(0x3642798750226111), CNST_LIMB(0x2df495ccaa57147b)},
        /*  50 */ {11, CNST_LIMB(0x2d5beb38dcd1394c), CNST_LIMB(0xb49a784bcd1b8afe), CNST_LIMB(0x43c33c1937564800), CNST_LIMB(0xe392010175ee5962)},
        /*  51 */ {11, CNST_LIMB(0x2d216f7943e2ba6a), CNST_LIMB(0xb5848226989d33c3), CNST_LIMB(0x54411b2441c3cd8b), CNST_LIMB(0x84eaf11b2fe7738e)},
        /*  52 */ {11, CNST_LIMB(0x2ce8a82efbb3ff2c), CNST_LIMB(0xb66a008e4788cbcd), CNST_LIMB(0x6851455acd400000), CNST_LIMB(0x3a1e3971e008995d)},
        /*  53 */ {11, CNST_LIMB(0x2cb17fea7ad7e332), CNST_LIMB(0xb74b1fd64e0753c6), CNST_LIMB(0x80a23b117c8feb6d), CNST_LIMB(0xfd7a462344ffce25)},
        /*  54 */ {11, CNST_LIMB(0x2c7be2b0cfa1ba50), CNST_LIMB(0xb82809d5be7072db), CNST_LIMB(0x9dff7d32d5dc1800), CNST_LIMB(0x9eca40b40ebcef8a)},
        /*  55 */ {11, CNST_LIMB(0x2c47bddba92d7463), CNST_LIMB(0xb900e6160002ccfe), CNST_LIMB(0xc155af6faeffe6a7), CNST_LIMB(0x52fa161a4a48e43d)},
        /*  56 */ {11, CNST_LIMB(0x2c14fffcaa8b131e), CNST_LIMB(0xb9d5d9fd5010b366), CNST_LIMB(0xebb7392e00000000), CNST_LIMB(0x1607a2cbacf930c1)},
        /*  57 */ {10, CNST_LIMB(0x2be398c3a38be053), CNST_LIMB(0xbaa708f58014d37c), CNST_LIMB(0x50633659656d971), CNST_LIMB(0x97a014f8e3be55f1)},
        /*  58 */ {10, CNST_LIMB(0x2bb378e758451068), CNST_LIMB(0xbb74948f5532da4b), CNST_LIMB(0x5fa8624c7fba400), CNST_LIMB(0x568df8b76cbf212c)},
        /*  59 */ {10, CNST_LIMB(0x2b8492108be5e5f7), CNST_LIMB(0xbc3e9ca2e1a05533), CNST_LIMB(0x717d9faa73c5679), CNST_LIMB(0x20ba7c4b4e6ef492)},
        /*  60 */ {10, CNST_LIMB(0x2b56d6c70d55481b), CNST_LIMB(0xbd053f6d26089673), CNST_LIMB(0x86430aac6100000), CNST_LIMB(0xe81ee46b9ef492f5)},
        /*  61 */ {10, CNST_LIMB(0x2b2a3a608c72ddd5), CNST_LIMB(0xbdc899ab3ff56c5e), CNST_LIMB(0x9e64d9944b57f29), CNST_LIMB(0x9dc0d10d51940416)},
        /*  62 */ {10, CNST_LIMB(0x2afeb0f1060c7e41), CNST_LIMB(0xbe88c6b3626a72aa), CNST_LIMB(0xba5ca5392cb0400), CNST_LIMB(0x5fa8ed2f450272a5)},
        /*  63 */ {10, CNST_LIMB(0x2ad42f3c9aca595c), CNST_LIMB(0xbf45e08bcf06554e), CNST_LIMB(0xdab2ce1d022cd81), CNST_LIMB(0x2ba9eb8c5e04e641)},
        /*  64 */ {10, CNST_LIMB(0x2aaaaaaaaaaaaaaa), CNST_LIMB(0xbfffffffffffffff), CNST_LIMB(0x6), CNST_LIMB(0x0)},
        /*  65 */ {10, CNST_LIMB(0x2a82193a13425883), CNST_LIMB(0xc0b73cb42e16914c), CNST_LIMB(0x12aeed5fd3e2d281), CNST_LIMB(0xb67759cc00287bf1)},
        /*  66 */ {10, CNST_LIMB(0x2a5a717672f66450), CNST_LIMB(0xc16bad3758efd873), CNST_LIMB(0x15c3da1572d50400), CNST_LIMB(0x78621feeb7f4ed33)},
        /*  67 */ {10, CNST_LIMB(0x2a33aa6e56d9c71c), CNST_LIMB(0xc21d6713f453f356), CNST_LIMB(0x194c05534f75ee29), CNST_LIMB(0x43d55b5f72943bc0)},
        /*  68 */ {10, CNST_LIMB(0x2a0dbbaa3bdfcea4), CNST_LIMB(0xc2cc7edf592262cf), CNST_LIMB(0x1d56299ada100000), CNST_LIMB(0x173decb64d1d4409)},
        /*  69 */ {10, CNST_LIMB(0x29e89d244eb4bfaf), CNST_LIMB(0xc379084815b5774c), CNST_LIMB(0x21f2a089a4ff4f79), CNST_LIMB(0xe29fb54fd6b6074f)},
        /*  70 */ {10, CNST_LIMB(0x29c44740d7db51e6), CNST_LIMB(0xc4231623369e78e5), CNST_LIMB(0x2733896c68d9a400), CNST_LIMB(0xa1f1f5c210d54e62)},
        /*  71 */ {10, CNST_LIMB(0x29a0b2c743b14d74), CNST_LIMB(0xc4caba789e2b8687), CNST_LIMB(0x2d2cf2c33b533c71), CNST_LIMB(0x6aac7f9bfafd57b2)},
        /*  72 */ {10, CNST_LIMB(0x297dd8dbb7c22a2d), CNST_LIMB(0xc570068e7ef5a1e7), CNST_LIMB(0x33f506e440000000), CNST_LIMB(0x3b563c2478b72ee2)},
        /*  73 */ {10, CNST_LIMB(0x295bb2f9285c8c1b), CNST_LIMB(0xc6130af40bc0ecbf), CNST_LIMB(0x3ba43bec1d062211), CNST_LIMB(0x12b536b574e92d1b)},
        /*  74 */ {10, CNST_LIMB(0x293a3aebe2be1c92), CNST_LIMB(0xc6b3d78b6d3b24fb), CNST_LIMB(0x4455872d8fd4e400), CNST_LIMB(0xdf86c03020404fa5)},
        /*  75 */ {10, CNST_LIMB(0x29196acc815ebd9f), CNST_LIMB(0xc7527b930c965bf2), CNST_LIMB(0x4e2694539f2f6c59), CNST_LIMB(0xa34adf02234eea8e)},
        /*  76 */ {10, CNST_LIMB(0x28f93cfb40f5c22a), CNST_LIMB(0xc7ef05ae409a0288), CNST_LIMB(0x5938006c18900000), CNST_LIMB(0x6f46eb8574eb59dd)},
        /*  77 */ {10, CNST_LIMB(0x28d9ac1badc64117), CNST_LIMB(0xc88983ed6985bae5), CNST_LIMB(0x65ad9912474aa649), CNST_LIMB(0x42459b481df47cec)},
        /*  78 */ {10, CNST_LIMB(0x28bab310a196b478), CNST_LIMB(0xc92203d587039cc1), CNST_LIMB(0x73ae9ff4241ec400), CNST_LIMB(0x1b424b95d80ca505)},
        /*  79 */ {10, CNST_LIMB(0x289c4cf88b774469), CNST_LIMB(0xc9b892675266f66c), CNST_LIMB(0x836612ee9c4ce1e1), CNST_LIMB(0xf2c1b982203a0dac)},
        /*  80 */ {10, CNST_LIMB(0x287e7529fb244e91), CNST_LIMB(0xca4d3c25e68dc57f), CNST_LIMB(0x9502f90000000000), CNST_LIMB(0xb7cdfd9d7bdbab7d)},
        /*  81 */ {10, CNST_LIMB(0x286127306a6a7a53), CNST_LIMB(0xcae00d1cfdeb43cf), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
        /*  82 */ {10, CNST_LIMB(0x28445ec93f792b1e), CNST_LIMB(0xcb7110e6ce866f2b), CNST_LIMB(0xbebf59a07dab4400), CNST_LIMB(0x57931eeaf85cf64f)},
        /*  83 */ {10, CNST_LIMB(0x282817e1038950fa), CNST_LIMB(0xcc0052b18b0e2a19), CNST_LIMB(0xd7540d4093bc3109), CNST_LIMB(0x305a944507c82f47)},
        /*  84 */ {10, CNST_LIMB(0x280c4e90c9ab1f45), CNST_LIMB(0xcc8ddd448f8b845a), CNST_LIMB(0xf2b96616f1900000), CNST_LIMB(0xe007ccc9c22781a)},
        /*  85 */ {9, CNST_LIMB(0x27f0ff1bc1ee87cd), CNST_LIMB(0xcd19bb053fb0284e), CNST_LIMB(0x336de62af2bca35), CNST_LIMB(0x3e92c42e000eeed4)},
        /*  86 */ {9, CNST_LIMB(0x27d625ecf571c340), CNST_LIMB(0xcda3f5fb9c415052), CNST_LIMB(0x39235ec33d49600), CNST_LIMB(0x1ebe59130db2795e)},
        /*  87 */ {9, CNST_LIMB(0x27bbbf95282fcd45), CNST_LIMB(0xce2c97d694adab3f), CNST_LIMB(0x3f674e539585a17), CNST_LIMB(0x268859e90f51b89)},
        /*  88 */ {9, CNST_LIMB(0x27a1c8c8ddaf84da), CNST_LIMB(0xceb3a9f01975077f), CNST_LIMB(0x4645b6958000000), CNST_LIMB(0xd24cde0463108cfa)},
        /*  89 */ {9, CNST_LIMB(0x27883e5e7df3f518), CNST_LIMB(0xcf393550f3aa6906), CNST_LIMB(0x4dcb74afbc49c19), CNST_LIMB(0xa536009f37adc383)},
        /*  90 */ {9, CNST_LIMB(0x276f1d4c9847e90e), CNST_LIMB(0xcfbd42b465836767), CNST_LIMB(0x56064e1d18d9a00), CNST_LIMB(0x7cea06ce1c9ace10)},
        /*  91 */ {9, CNST_LIMB(0x275662a841b30191), CNST_LIMB(0xd03fda8b97997f33), CNST_LIMB(0x5f04fe2cd8a39fb), CNST_LIMB(0x58db032e72e8ba43)},
        /*  92 */ {9, CNST_LIMB(0x273e0ba38d15a47b), CNST_LIMB(0xd0c10500d63aa658), CNST_LIMB(0x68d74421f5c0000), CNST_LIMB(0x388cc17cae105447)},
        /*  93 */ {9, CNST_LIMB(0x2726158c1b13cf03), CNST_LIMB(0xd140c9faa1e5439e), CNST_LIMB(0x738df1f6ab4827d), CNST_LIMB(0x1b92672857620ce0)},
        /*  94 */ {9, CNST_LIMB(0x270e7dc9c01d8e9b), CNST_LIMB(0xd1bf311e95d00de3), CNST_LIMB(0x7f3afbc9cfb5e00), CNST_LIMB(0x18c6a9575c2ade4)},
        /*  95 */ {9, CNST_LIMB(0x26f741dd3f070d61), CNST_LIMB(0xd23c41d42727c808), CNST_LIMB(0x8bf187fba88f35f), CNST_LIMB(0xd44da7da8e44b24f)},
        /*  96 */ {9, CNST_LIMB(0x26e05f5f16c2159e), CNST_LIMB(0xd2b803473f7ad0f3), CNST_LIMB(0x99c600000000000), CNST_LIMB(0xaa2f78f1b4cc6794)},
        /*  97 */ {9, CNST_LIMB(0x26c9d3fe61e80598), CNST_LIMB(0xd3327c6ab49ca6c8), CNST_LIMB(0xa8ce21eb6531361), CNST_LIMB(0x843c067d091ee4cc)},
        /*  98 */ {9, CNST_LIMB(0x26b39d7fc6ddab08), CNST_LIMB(0xd3abb3faa02166cc), CNST_LIMB(0xb92112c1a0b6200), CNST_LIMB(0x62005e1e913356e3)},
        /*  99 */ {9, CNST_LIMB(0x269db9bc7772a5cc), CNST_LIMB(0xd423b07e986aa967), CNST_LIMB(0xcad7718b8747c43), CNST_LIMB(0x4316eed01dedd518)},
        /* 100 */ {9, CNST_LIMB(0x268826a13ef3fde6), CNST_LIMB(0xd49a784bcd1b8afe), CNST_LIMB(0xde0b6b3a7640000), CNST_LIMB(0x2725dd1d243aba0e)},
        /* 101 */ {9, CNST_LIMB(0x2672e22d9dbdbd9f), CNST_LIMB(0xd510118708a8f8dd), CNST_LIMB(0xf2d8cf5fe6d74c5), CNST_LIMB(0xddd9057c24cb54f)},
        /* 102 */ {9, CNST_LIMB(0x265dea72f169cc99), CNST_LIMB(0xd5848226989d33c3), CNST_LIMB(0x1095d25bfa712600), CNST_LIMB(0xedeee175a736d2a1)},
        /* 103 */ {9, CNST_LIMB(0x26493d93a8cb2514), CNST_LIMB(0xd5f7cff41e09aeb8), CNST_LIMB(0x121b7c4c3698faa7), CNST_LIMB(0xc4699f3df8b6b328)},
        /* 104 */ {9, CNST_LIMB(0x2634d9c282f3ef82), CNST_LIMB(0xd66a008e4788cbcd), CNST_LIMB(0x13c09e8d68000000), CNST_LIMB(0x9ebbe7d859cb5a7c)},
        /* 105 */ {9, CNST_LIMB(0x2620bd41d8933adc), CNST_LIMB(0xd6db196a761949d9), CNST_LIMB(0x15876ccb0b709ca9), CNST_LIMB(0x7c828b9887eb2179)},
        /* 106 */ {9, CNST_LIMB(0x260ce662ef04088a), CNST_LIMB(0xd74b1fd64e0753c6), CNST_LIMB(0x17723c2976da2a00), CNST_LIMB(0x5d652ab99001adcf)},
        /* 107 */ {9, CNST_LIMB(0x25f95385547353fd), CNST_LIMB(0xd7ba18f93502e409), CNST_LIMB(0x198384e9c259048b), CNST_LIMB(0x4114f1754e5d7b32)},
        /* 108 */ {9, CNST_LIMB(0x25e60316448db8e1), CNST_LIMB(0xd82809d5be7072db), CNST_LIMB(0x1bbde41dfeec0000), CNST_LIMB(0x274b7c902f7e0188)},
        /* 109 */ {9, CNST_LIMB(0x25d2f390152f74f5), CNST_LIMB(0xd894f74b06ef8b40), CNST_LIMB(0x1e241d6e3337910d), CNST_LIMB(0xfc9e0fbb32e210c)},
        /* 110 */ {9, CNST_LIMB(0x25c02379aa9ad043), CNST_LIMB(0xd900e6160002ccfe), CNST_LIMB(0x20b91cee9901ee00), CNST_LIMB(0xf4afa3e594f8ea1f)},
        /* 111 */ {9, CNST_LIMB(0x25ad9165f2c18907), CNST_LIMB(0xd96bdad2acb5f5ef), CNST_LIMB(0x237ff9079863dfef), CNST_LIMB(0xcd85c32e9e4437b0)},
        /* 112 */ {9, CNST_LIMB(0x259b3bf36735c90c), CNST_LIMB(0xd9d5d9fd5010b366), CNST_LIMB(0x267bf47000000000), CNST_LIMB(0xa9bbb147e0dd92a8)},
        /* 113 */ {9, CNST_LIMB(0x258921cb955e7693), CNST_LIMB(0xda3ee7f38e181ed0), CNST_LIMB(0x29b08039fbeda7f1), CNST_LIMB(0x8900447b70e8eb82)},
        /* 114 */ {9, CNST_LIMB(0x257741a2ac9170af), CNST_LIMB(0xdaa708f58014d37c), CNST_LIMB(0x2d213df34f65f200), CNST_LIMB(0x6b0a92adaad5848a)},
        /* 115 */ {9, CNST_LIMB(0x25659a3711bc827d), CNST_LIMB(0xdb0e4126bcc86bd7), CNST_LIMB(0x30d201d957a7c2d3), CNST_LIMB(0x4f990ad8740f0ee5)},
        /* 116 */ {9, CNST_LIMB(0x25542a50f84b9c39), CNST_LIMB(0xdb74948f5532da4b), CNST_LIMB(0x34c6d52160f40000), CNST_LIMB(0x3670a9663a8d3610)},
        /* 117 */ {9, CNST_LIMB(0x2542f0c20000377d), CNST_LIMB(0xdbda071cc67e6db5), CNST_LIMB(0x3903f855d8f4c755), CNST_LIMB(0x1f5c44188057be3c)},
        /* 118 */ {9, CNST_LIMB(0x2531ec64d772bd64), CNST_LIMB(0xdc3e9ca2e1a05533), CNST_LIMB(0x3d8de5c8ec59b600), CNST_LIMB(0xa2bea956c4e4977)},
        /* 119 */ {9, CNST_LIMB(0x25211c1ce2fb5a6e), CNST_LIMB(0xdca258dca9331635), CNST_LIMB(0x4269541d1ff01337), CNST_LIMB(0xed68b23033c3637e)},
        /* 120 */ {9, CNST_LIMB(0x25107ed5e7c3ec3b), CNST_LIMB(0xdd053f6d26089673), CNST_LIMB(0x479b38e478000000), CNST_LIMB(0xc99cf624e50549c5)},
        /* 121 */ {9, CNST_LIMB(0x25001383bac8a744), CNST_LIMB(0xdd6753e032ea0efe), CNST_LIMB(0x4d28cb56c33fa539), CNST_LIMB(0xa8adf7ae45e7577b)},
        /* 122 */ {9, CNST_LIMB(0x24efd921f390bce3), CNST_LIMB(0xddc899ab3ff56c5e), CNST_LIMB(0x5317871fa13aba00), CNST_LIMB(0x8a5bc740b1c113e5)},
        /* 123 */ {9, CNST_LIMB(0x24dfceb3a26bb203), CNST_LIMB(0xde29142e0e01401f), CNST_LIMB(0x596d2f44de9fa71b), CNST_LIMB(0x6e6c7efb81cfbb9b)},
        /* 124 */ {9, CNST_LIMB(0x24cff3430a0341a7), CNST_LIMB(0xde88c6b3626a72aa), CNST_LIMB(0x602fd125c47c0000), CNST_LIMB(0x54aba5c5cada5f10)},
        /* 125 */ {9, CNST_LIMB(0x24c045e15c149931), CNST_LIMB(0xdee7b471b3a9507d), CNST_LIMB(0x6765c793fa10079d), CNST_LIMB(0x3ce9a36f23c0fc90)},
        /* 126 */ {9, CNST_LIMB(0x24b0c5a679267ae2), CNST_LIMB(0xdf45e08bcf06554e), CNST_LIMB(0x6f15be069b847e00), CNST_LIMB(0x26fb43de2c8cd2a8)},
        /* 127 */ {9, CNST_LIMB(0x24a171b0b31461c8), CNST_LIMB(0xdfa34e1177c23362), CNST_LIMB(0x7746b3e82a77047f), CNST_LIMB(0x12b94793db8486a1)},
        /* 128 */ {9, CNST_LIMB(0x2492492492492492), CNST_LIMB(0xdfffffffffffffff), CNST_LIMB(0x7), CNST_LIMB(0x0)},
        /* 129 */ {9, CNST_LIMB(0x24834b2c9d85cdfe), CNST_LIMB(0xe05bf942dbbc2145), CNST_LIMB(0x894953f7ea890481), CNST_LIMB(0xdd5deca404c0156d)},
        /* 130 */ {9, CNST_LIMB(0x247476f924137501), CNST_LIMB(0xe0b73cb42e16914c), CNST_LIMB(0x932abffea4848200), CNST_LIMB(0xbd51373330291de0)},
        /* 131 */ {9, CNST_LIMB(0x2465cbc00a40cec0), CNST_LIMB(0xe111cd1d5133412e), CNST_LIMB(0x9dacb687d3d6a163), CNST_LIMB(0x9fa4025d66f23085)},
        /* 132 */ {9, CNST_LIMB(0x245748bc980e0427), CNST_LIMB(0xe16bad3758efd873), CNST_LIMB(0xa8d8102a44840000), CNST_LIMB(0x842530ee2db4949d)},
        /* 133 */ {9, CNST_LIMB(0x2448ed2f49eb0633), CNST_LIMB(0xe1c4dfab90aab5ef), CNST_LIMB(0xb4b60f9d140541e5), CNST_LIMB(0x6aa7f2766b03dc25)},
        /* 134 */ {9, CNST_LIMB(0x243ab85da36e3167), CNST_LIMB(0xe21d6713f453f356), CNST_LIMB(0xc15065d4856e4600), CNST_LIMB(0x53035ba7ebf32e8d)},
        /* 135 */ {9, CNST_LIMB(0x242ca99203ea8c18), CNST_LIMB(0xe27545fba4fe385a), CNST_LIMB(0xceb1363f396d23c7), CNST_LIMB(0x3d12091fc9fb4914)},
        /* 136 */ {9, CNST_LIMB(0x241ec01b7cce4ea0), CNST_LIMB(0xe2cc7edf592262cf), CNST_LIMB(0xdce31b2488000000), CNST_LIMB(0x28b1cb81b1ef1849)},
        /* 137 */ {9, CNST_LIMB(0x2410fb4da9b3b0fc), CNST_LIMB(0xe323142dc8c66b55), CNST_LIMB(0xebf12a24bca135c9), CNST_LIMB(0x15c35be67ae3e2c9)},
        /* 138 */ {9, CNST_LIMB(0x24035a808a0f315e), CNST_LIMB(0xe379084815b5774c), CNST_LIMB(0xfbe6f8dbf88f4a00), CNST_LIMB(0x42a17bd09be1ff0)},
        /* 139 */ {8, CNST_LIMB(0x23f5dd105c67ab9d), CNST_LIMB(0xe3ce5d822ff4b643), CNST_LIMB(0x1ef156c084ce761), CNST_LIMB(0x8bf461f03cf0bbf)},
        /* 140 */ {8, CNST_LIMB(0x23e8825d7b05abb1), CNST_LIMB(0xe4231623369e78e5), CNST_LIMB(0x20c4e3b94a10000), CNST_LIMB(0xf3fbb43f68a32d05)},
        /* 141 */ {8, CNST_LIMB(0x23db49cc3a0866fe), CNST_LIMB(0xe4773465d54aded7), CNST_LIMB(0x22b0695a08ba421), CNST_LIMB(0xd84f44c48564dc19)},
        /* 142 */ {8, CNST_LIMB(0x23ce32c4c6cfb9f5), CNST_LIMB(0xe4caba789e2b8687), CNST_LIMB(0x24b4f35d7a4c100), CNST_LIMB(0xbe58ebcce7956abe)},
        /* 143 */ {8, CNST_LIMB(0x23c13cb308ab6ab7), CNST_LIMB(0xe51daa7e60fdd34c), CNST_LIMB(0x26d397284975781), CNST_LIMB(0xa5fac463c7c134b7)},
        /* 144 */ {8, CNST_LIMB(0x23b4670682c0c709), CNST_LIMB(0xe570068e7ef5a1e7), CNST_LIMB(0x290d74100000000), CNST_LIMB(0x8f19241e28c7d757)},
        /* 145 */ {8, CNST_LIMB(0x23a7b13237187c8b), CNST_LIMB(0xe5c1d0b53bc09fca), CNST_LIMB(0x2b63b3a37866081), CNST_LIMB(0x799a6d046c0ae1ae)},
        /* 146 */ {8, CNST_LIMB(0x239b1aac8ac74728), CNST_LIMB(0xe6130af40bc0ecbf), CNST_LIMB(0x2dd789f4d894100), CNST_LIMB(0x6566e37d746a9e40)},
        /* 147 */ {8, CNST_LIMB(0x238ea2ef2b24c379), CNST_LIMB(0xe663b741df9c37c0), CNST_LIMB(0x306a35e51b58721), CNST_LIMB(0x526887dbfb5f788f)},
        /* 148 */ {8, CNST_LIMB(0x23824976f4045a26), CNST_LIMB(0xe6b3d78b6d3b24fb), CNST_LIMB(0x331d01712e10000), CNST_LIMB(0x408af3382b8efd3d)},
        /* 149 */ {8, CNST_LIMB(0x23760dc3d6e4d729), CNST_LIMB(0xe7036db376537b90), CNST_LIMB(0x35f14200a827c61), CNST_LIMB(0x2fbb374806ec05f1)},
        /* 150 */ {8, CNST_LIMB(0x2369ef58c30bd43e), CNST_LIMB(0xe7527b930c965bf2), CNST_LIMB(0x38e858b62216100), CNST_LIMB(0x1fe7c0f0afce87fe)},
        /* 151 */ {8, CNST_LIMB(0x235dedbb8e82aa1c), CNST_LIMB(0xe7a102f9d39a9331), CNST_LIMB(0x3c03b2c13176a41), CNST_LIMB(0x11003d517540d32e)},
        /* 152 */ {8, CNST_LIMB(0x23520874dfeb1ffd), CNST_LIMB(0xe7ef05ae409a0288), CNST_LIMB(0x3f44c9b21000000), CNST_LIMB(0x2f5810f98eff0dc)},
        /* 153 */ {8, CNST_LIMB(0x23463f1019228dd7), CNST_LIMB(0xe83c856dd81804b7), CNST_LIMB(0x42ad23cef3113c1), CNST_LIMB(0xeb72e35e7840d910)},
        /* 154 */ {8, CNST_LIMB(0x233a911b42aa9b3c), CNST_LIMB(0xe88983ed6985bae5), CNST_LIMB(0x463e546b19a2100), CNST_LIMB(0xd27de19593dc3614)},
        /* 155 */ {8, CNST_LIMB(0x232efe26f7cf33f9), CNST_LIMB(0xe8d602d948f83829), CNST_LIMB(0x49f9fc3f96684e1), CNST_LIMB(0xbaf391fd3e5e6fc2)},
        /* 156 */ {8, CNST_LIMB(0x232385c65381b485), CNST_LIMB(0xe92203d587039cc1), CNST_LIMB(0x4de1c9c5dc10000), CNST_LIMB(0xa4bd38c55228c81d)},
        /* 157 */ {8, CNST_LIMB(0x2318278edde1b39b), CNST_LIMB(0xe96d887e26cd57b7), CNST_LIMB(0x51f77994116d2a1), CNST_LIMB(0x8fc5a8de8e1de782)},
        /* 158 */ {8, CNST_LIMB(0x230ce3187a6c2be9), CNST_LIMB(0xe9b892675266f66c), CNST_LIMB(0x563cd6bb3398100), CNST_LIMB(0x7bf9265bea9d3a3b)},
        /* 159 */ {8, CNST_LIMB(0x2301b7fd56ca21bb), CNST_LIMB(0xea03231d8d8224ba), CNST_LIMB(0x5ab3bb270beeb01), CNST_LIMB(0x69454b325983dccd)},
        /* 160 */ {8, CNST_LIMB(0x22f6a5d9da38341c), CNST_LIMB(0xea4d3c25e68dc57f), CNST_LIMB(0x5f5e10000000000), CNST_LIMB(0x5798ee2308c39df9)},
        /* 161 */ {8, CNST_LIMB(0x22ebac4c9580d89f), CNST_LIMB(0xea96defe264b59be), CNST_LIMB(0x643dce0ec16f501), CNST_LIMB(0x46e40ba0fa66a753)},
        /* 162 */ {8, CNST_LIMB(0x22e0caf633834beb), CNST_LIMB(0xeae00d1cfdeb43cf), CNST_LIMB(0x6954fe21e3e8100), CNST_LIMB(0x3717b0870b0db3a7)},
        /* 163 */ {8, CNST_LIMB(0x22d601796a418886), CNST_LIMB(0xeb28c7f233bdd372), CNST_LIMB(0x6ea5b9755f440a1), CNST_LIMB(0x2825e6775d11cdeb)},
        /* 164 */ {8, CNST_LIMB(0x22cb4f7aec6fd8b4), CNST_LIMB(0xeb7110e6ce866f2b), CNST_LIMB(0x74322a1c0410000), CNST_LIMB(0x1a01a1c09d1b4dac)},
        /* 165 */ {8, CNST_LIMB(0x22c0b4a15b80d83e), CNST_LIMB(0xebb8e95d3f7d9df2), CNST_LIMB(0x79fc8b6ae8a46e1), CNST_LIMB(0xc9eb0a8bebc8f3e)},
        /* 166 */ {8, CNST_LIMB(0x22b630953a28f77a), CNST_LIMB(0xec0052b18b0e2a19), CNST_LIMB(0x80072a66d512100), CNST_LIMB(0xffe357ff59e6a004)},
        /* 167 */ {8, CNST_LIMB(0x22abc300df54ca7c), CNST_LIMB(0xec474e39705912d2), CNST_LIMB(0x86546633b42b9c1), CNST_LIMB(0xe7dfd1be05fa61a8)},
        /* 168 */ {8, CNST_LIMB(0x22a16b90698da5d2), CNST_LIMB(0xec8ddd448f8b845a), CNST_LIMB(0x8ce6b0861000000), CNST_LIMB(0xd11ed6fc78f760e5)},
        /* 169 */ {8, CNST_LIMB(0x229729f1b2c83ded), CNST_LIMB(0xecd4011c8f11979a), CNST_LIMB(0x93c08e16a022441), CNST_LIMB(0xbb8db609dd29ebfe)},
        /* 170 */ {8, CNST_LIMB(0x228cfdd444992f78), CNST_LIMB(0xed19bb053fb0284e), CNST_LIMB(0x9ae49717f026100), CNST_LIMB(0xa71aec8d1813d532)},
        /* 171 */ {8, CNST_LIMB(0x2282e6e94ccb8588), CNST_LIMB(0xed5f0c3cbf8fa470), CNST_LIMB(0xa25577ae24c1a61), CNST_LIMB(0x93b612a9f20fbc02)},
        /* 172 */ {8, CNST_LIMB(0x2278e4e392557ecf), CNST_LIMB(0xeda3f5fb9c415052), CNST_LIMB(0xaa15f068e610000), CNST_LIMB(0x814fc7b19a67d317)},
        /* 173 */ {8, CNST_LIMB(0x226ef7776aa7fd29), CNST_LIMB(0xede87974f3c81855), CNST_LIMB(0xb228d6bf7577921), CNST_LIMB(0x6fd9a03f2e0a4b7c)},
        /* 174 */ {8, CNST_LIMB(0x22651e5aaf5532d0), CNST_LIMB(0xee2c97d694adab3f), CNST_LIMB(0xba91158ef5c4100), CNST_LIMB(0x5f4615a38d0d316e)},
        /* 175 */ {8, CNST_LIMB(0x225b5944b40b4694), CNST_LIMB(0xee7052491d2c3e64), CNST_LIMB(0xc351ad9aec0b681), CNST_LIMB(0x4f8876863479a286)},
        /* 176 */ {8, CNST_LIMB(0x2251a7ee3cdfcca5), CNST_LIMB(0xeeb3a9f01975077f), CNST_LIMB(0xcc6db6100000000), CNST_LIMB(0x4094d8a3041b60eb)},
        /* 177 */ {8, CNST_LIMB(0x22480a1174e913d9), CNST_LIMB(0xeef69fea211b2627), CNST_LIMB(0xd5e85d09025c181), CNST_LIMB(0x32600b8ed883a09b)},
        /* 178 */ {8, CNST_LIMB(0x223e7f69e522683c), CNST_LIMB(0xef393550f3aa6906), CNST_LIMB(0xdfc4e816401c100), CNST_LIMB(0x24df8c6eb4b6d1f1)},
        /* 179 */ {8, CNST_LIMB(0x223507b46b988abe), CNST_LIMB(0xef7b6b399471103e), CNST_LIMB(0xea06b4c72947221), CNST_LIMB(0x18097a8ee151acef)},
        /* 180 */ {8, CNST_LIMB(0x222ba2af32dbbb9e), CNST_LIMB(0xefbd42b465836767), CNST_LIMB(0xf4b139365210000), CNST_LIMB(0xbd48cc8ec1cd8e3)},
        /* 181 */ {8, CNST_LIMB(0x22225019a9b4d16c), CNST_LIMB(0xeffebccd41ffcd5c), CNST_LIMB(0xffc80497d520961), CNST_LIMB(0x3807a8d67485fb)},
        /* 182 */ {8, CNST_LIMB(0x22190fb47b1af172), CNST_LIMB(0xf03fda8b97997f33), CNST_LIMB(0x10b4ebfca1dee100), CNST_LIMB(0xea5768860b62e8d8)},
        /* 183 */ {8, CNST_LIMB(0x220fe14186679801), CNST_LIMB(0xf0809cf27f703d52), CNST_LIMB(0x117492de921fc141), CNST_LIMB(0xd54faf5b635c5005)},
        /* 184 */ {8, CNST_LIMB(0x2206c483d7c6b786), CNST_LIMB(0xf0c10500d63aa658), CNST_LIMB(0x123bb2ce41000000), CNST_LIMB(0xc14a56233a377926)},
        /* 185 */ {8, CNST_LIMB(0x21fdb93fa0e0ccc5), CNST_LIMB(0xf10113b153c8ea7b), CNST_LIMB(0x130a8b6157bdecc1), CNST_LIMB(0xae39a88db7cd329f)},
        /* 186 */ {8, CNST_LIMB(0x21f4bf3a31bcdcaa), CNST_LIMB(0xf140c9faa1e5439e), CNST_LIMB(0x13e15dede0e8a100), CNST_LIMB(0x9c10bde69efa7ab6)},
        /* 187 */ {8, CNST_LIMB(0x21ebd639f1d86584), CNST_LIMB(0xf18028cf72976a4e), CNST_LIMB(0x14c06d941c0ca7e1), CNST_LIMB(0x8ac36c42a2836497)},
        /* 188 */ {8, CNST_LIMB(0x21e2fe06597361a6), CNST_LIMB(0xf1bf311e95d00de3), CNST_LIMB(0x15a7ff487a810000), CNST_LIMB(0x7a463c8b84f5ef67)},
        /* 189 */ {8, CNST_LIMB(0x21da3667eb0e8ccb), CNST_LIMB(0xf1fde3d30e812642), CNST_LIMB(0x169859ddc5c697a1), CNST_LIMB(0x6a8e5f5ad090fd4b)},
        /* 190 */ {8, CNST_LIMB(0x21d17f282d1a300e), CNST_LIMB(0xf23c41d42727c808), CNST_LIMB(0x1791c60f6fed0100), CNST_LIMB(0x5b91a2943596fc56)},
        /* 191 */ {8, CNST_LIMB(0x21c8d811a3d3c9e1), CNST_LIMB(0xf27a4c0585cbf805), CNST_LIMB(0x18948e8c0e6fba01), CNST_LIMB(0x4d4667b1c468e8f0)},
        /* 192 */ {8, CNST_LIMB(0x21c040efcb50f858), CNST_LIMB(0xf2b803473f7ad0f3), CNST_LIMB(0x19a1000000000000), CNST_LIMB(0x3fa39ab547994daf)},
        /* 193 */ {8, CNST_LIMB(0x21b7b98f11b61c1a), CNST_LIMB(0xf2f56875eb3f2614), CNST_LIMB(0x1ab769203dafc601), CNST_LIMB(0x32a0a9b2faee1e2a)},
        /* 194 */ {8, CNST_LIMB(0x21af41bcd19739ba), CNST_LIMB(0xf3327c6ab49ca6c8), CNST_LIMB(0x1bd81ab557f30100), CNST_LIMB(0x26357ceac0e96962)},
        /* 195 */ {8, CNST_LIMB(0x21a6d9474c81adf0), CNST_LIMB(0xf36f3ffb6d916240), CNST_LIMB(0x1d0367a69fed1ba1), CNST_LIMB(0x1a5a6f65caa5859e)},
        /* 196 */ {8, CNST_LIMB(0x219e7ffda5ad572a), CNST_LIMB(0xf3abb3faa02166cc), CNST_LIMB(0x1e39a5057d810000), CNST_LIMB(0xf08480f672b4e86)},
        /* 197 */ {8, CNST_LIMB(0x219635afdcd3e46d), CNST_LIMB(0xf3e7d9379f70166a), CNST_LIMB(0x1f7b2a18f29ac3e1), CNST_LIMB(0x4383340615612ca)},
        /* 198 */ {8, CNST_LIMB(0x218dfa2ec92d0643), CNST_LIMB(0xf423b07e986aa967), CNST_LIMB(0x20c850694c2aa100), CNST_LIMB(0xf3c77969ee4be5a2)},
        /* 199 */ {8, CNST_LIMB(0x2185cd4c148e4ae2), CNST_LIMB(0xf45f3a98a20738a4), CNST_LIMB(0x222173cc014980c1), CNST_LIMB(0xe00993cc187c5ec9)},
        /* 200 */ {8, CNST_LIMB(0x217daeda36ad7a5c), CNST_LIMB(0xf49a784bcd1b8afe), CNST_LIMB(0x2386f26fc1000000), CNST_LIMB(0xcd2b297d889bc2b6)},
        /* 201 */ {8, CNST_LIMB(0x21759eac708452fe), CNST_LIMB(0xf4d56a5b33cec44a), CNST_LIMB(0x24f92ce8af296d41), CNST_LIMB(0xbb214d5064862b22)},
        /* 202 */ {8, CNST_LIMB(0x216d9c96c7d490d4), CNST_LIMB(0xf510118708a8f8dd), CNST_LIMB(0x2678863cd0ece100), CNST_LIMB(0xa9e1a7ca7ea10e20)},
        /* 203 */ {8, CNST_LIMB(0x2165a86e02cb358c), CNST_LIMB(0xf54a6e8ca5438db1), CNST_LIMB(0x280563f0a9472d61), CNST_LIMB(0x99626e72b39ea0cf)},
        /* 204 */ {8, CNST_LIMB(0x215dc207a3c20fdf), CNST_LIMB(0xf5848226989d33c3), CNST_LIMB(0x29a02e1406210000), CNST_LIMB(0x899a5ba9c13fafd9)},
        /* 205 */ {8, CNST_LIMB(0x2155e939e51e8b37), CNST_LIMB(0xf5be4d0cb51434aa), CNST_LIMB(0x2b494f4efe6d2e21), CNST_LIMB(0x7a80a705391e96ff)},
        /* 206 */ {8, CNST_LIMB(0x214e1ddbb54cd933), CNST_LIMB(0xf5f7cff41e09aeb8), CNST_LIMB(0x2d0134ef21cbc100), CNST_LIMB(0x6c0cfe23de23042a)},
        /* 207 */ {8, CNST_LIMB(0x21465fc4b2d68f98), CNST_LIMB(0xf6310b8f55304840), CNST_LIMB(0x2ec84ef4da2ef581), CNST_LIMB(0x5e377df359c944dd)},
        /* 208 */ {8, CNST_LIMB(0x213eaecd2893dd60), CNST_LIMB(0xf66a008e4788cbcd), CNST_LIMB(0x309f102100000000), CNST_LIMB(0x50f8ac5fc8f53985)},
        /* 209 */ {8, CNST_LIMB(0x21370ace09f681c6), CNST_LIMB(0xf6a2af9e5a0f0a08), CNST_LIMB(0x3285ee02a1420281), CNST_LIMB(0x44497266278e35b7)},
        /* 210 */ {8, CNST_LIMB(0x212f73a0ef6db7cb), CNST_LIMB(0xf6db196a761949d9), CNST_LIMB(0x347d6104fc324100), CNST_LIMB(0x382316831f7ee175)},
        /* 211 */ {8, CNST_LIMB(0x2127e92012e25004), CNST_LIMB(0xf7133e9b156c7be5), CNST_LIMB(0x3685e47dade53d21), CNST_LIMB(0x2c7f377833b8946e)},
        /* 212 */ {8, CNST_LIMB(0x21206b264c4a39a7), CNST_LIMB(0xf74b1fd64e0753c6), CNST_LIMB(0x389ff6bb15610000), CNST_LIMB(0x2157c761ab4163ef)},
        /* 213 */ {8, CNST_LIMB(0x2118f98f0e52c28f), CNST_LIMB(0xf782bdbfdda6577b), CNST_LIMB(0x3acc1912ebb57661), CNST_LIMB(0x16a7071803cc49a9)},
        /* 214 */ {8, CNST_LIMB(0x211194366320dc66), CNST_LIMB(0xf7ba18f93502e409), CNST_LIMB(0x3d0acff111946100), CNST_LIMB(0xc6781d80f8224fc)},
        /* 215 */ {8, CNST_LIMB(0x210a3af8e926bb78), CNST_LIMB(0xf7f1322182cf15d1), CNST_LIMB(0x3f5ca2e692eaf841), CNST_LIMB(0x294092d370a900b)},
        /* 216 */ {8, CNST_LIMB(0x2102edb3d00e29a6), CNST_LIMB(0xf82809d5be7072db), CNST_LIMB(0x41c21cb8e1000000), CNST_LIMB(0xf24f62335024a295)},
        /* 217 */ {8, CNST_LIMB(0x20fbac44d5b6edc2), CNST_LIMB(0xf85ea0b0b27b2610), CNST_LIMB(0x443bcb714399a5c1), CNST_LIMB(0xe03b98f103fad6d2)},
        /* 218 */ {8, CNST_LIMB(0x20f4768a4348ad08), CNST_LIMB(0xf894f74b06ef8b40), CNST_LIMB(0x46ca406c81af2100), CNST_LIMB(0xcee3d32cad2a9049)},
        /* 219 */ {8, CNST_LIMB(0x20ed4c62ea57b1f0), CNST_LIMB(0xf8cb0e3b4b3bbdb3), CNST_LIMB(0x496e106ac22aaae1), CNST_LIMB(0xbe3f9df9277fdada)},
        /* 220 */ {8, CNST_LIMB(0x20e62dae221c087a), CNST_LIMB(0xf900e6160002ccfe), CNST_LIMB(0x4c27d39fa5410000), CNST_LIMB(0xae46f0d94c05e933)},
        /* 221 */ {8, CNST_LIMB(0x20df1a4bc4ba6525), CNST_LIMB(0xf9367f6da0ab2e9c), CNST_LIMB(0x4ef825c296e43ca1), CNST_LIMB(0x9ef2280fb437a33d)},
        /* 222 */ {8, CNST_LIMB(0x20d8121c2c9e506e), CNST_LIMB(0xf96bdad2acb5f5ef), CNST_LIMB(0x51dfa61f5ad88100), CNST_LIMB(0x9039ff426d3f284b)},
        /* 223 */ {8, CNST_LIMB(0x20d1150031e51549), CNST_LIMB(0xf9a0f8d3b0e04fde), CNST_LIMB(0x54def7a6d2f16901), CNST_LIMB(0x82178c6d6b51f8f4)},
        /* 224 */ {8, CNST_LIMB(0x20ca22d927d8f54d), CNST_LIMB(0xf9d5d9fd5010b366), CNST_LIMB(0x57f6c10000000000), CNST_LIMB(0x74843b1ee4c1e053)},
        /* 225 */ {8, CNST_LIMB(0x20c33b88da7c29aa), CNST_LIMB(0xfa0a7eda4c112ce6), CNST_LIMB(0x5b27ac993df97701), CNST_LIMB(0x6779c7f90dc42f48)},
        /* 226 */ {8, CNST_LIMB(0x20bc5ef18c233bdf), CNST_LIMB(0xfa3ee7f38e181ed0), CNST_LIMB(0x5e7268b9bbdf8100), CNST_LIMB(0x5af23c74f9ad9fe9)},
        /* 227 */ {8, CNST_LIMB(0x20b58cf5f31e4526), CNST_LIMB(0xfa7315d02f20c7bd), CNST_LIMB(0x61d7a7932ff3d6a1), CNST_LIMB(0x4ee7eae2acdc617e)},
        /* 228 */ {8, CNST_LIMB(0x20aec5793770a74d), CNST_LIMB(0xfaa708f58014d37c), CNST_LIMB(0x65581f53c8c10000), CNST_LIMB(0x43556aa2ac262a0b)},
        /* 229 */ {8, CNST_LIMB(0x20a8085ef096d530), CNST_LIMB(0xfadac1e711c832d1), CNST_LIMB(0x68f48a385b8320e1), CNST_LIMB(0x3835949593b8ddd1)},
        /* 230 */ {8, CNST_LIMB(0x20a1558b2359c4b1), CNST_LIMB(0xfb0e4126bcc86bd7), CNST_LIMB(0x6cada69ed07c2100), CNST_LIMB(0x2d837fbe78458762)},
        /* 231 */ {8, CNST_LIMB(0x209aace23fafa72e), CNST_LIMB(0xfb418734a9008bd9), CNST_LIMB(0x70843718cdbf27c1), CNST_LIMB(0x233a7e150a54a555)},
        /* 232 */ {8, CNST_LIMB(0x20940e491ea988d7), CNST_LIMB(0xfb74948f5532da4b), CNST_LIMB(0x7479027ea1000000), CNST_LIMB(0x19561984a50ff8fe)},
        /* 233 */ {8, CNST_LIMB(0x208d79a5006d7a47), CNST_LIMB(0xfba769b39e49640e), CNST_LIMB(0x788cd40268f39641), CNST_LIMB(0xfd211159fe3490f)},
        /* 234 */ {8, CNST_LIMB(0x2086eedb8a3cead3), CNST_LIMB(0xfbda071cc67e6db5), CNST_LIMB(0x7cc07b437ecf6100), CNST_LIMB(0x6aa563e655033e3)},
        /* 235 */ {8, CNST_LIMB(0x20806dd2c486dcc6), CNST_LIMB(0xfc0c6d447c5dd362), CNST_LIMB(0x8114cc6220762061), CNST_LIMB(0xfbb614b3f2d3b14c)},
        /* 236 */ {8, CNST_LIMB(0x2079f67119059fae), CNST_LIMB(0xfc3e9ca2e1a05533), CNST_LIMB(0x858aa0135be10000), CNST_LIMB(0xeac0f8837fb05773)},
        /* 237 */ {8, CNST_LIMB(0x2073889d50e7bf63), CNST_LIMB(0xfc7095ae91e1c760), CNST_LIMB(0x8a22d3b53c54c321), CNST_LIMB(0xda6e4c10e8615ca5)},
        /* 238 */ {8, CNST_LIMB(0x206d243e9303d929), CNST_LIMB(0xfca258dca9331635), CNST_LIMB(0x8ede496339f34100), CNST_LIMB(0xcab755a8d01fa67f)},
        /* 239 */ {8, CNST_LIMB(0x2066c93c62170aa8), CNST_LIMB(0xfcd3e6a0ca8906c2), CNST_LIMB(0x93bde80aec3a1481), CNST_LIMB(0xbb95a9ae71aa3e0c)},
        /* 240 */ {8, CNST_LIMB(0x2060777e9b0db0f6), CNST_LIMB(0xfd053f6d26089673), CNST_LIMB(0x98c29b8100000000), CNST_LIMB(0xad0326c296b4f529)},
        /* 241 */ {8, CNST_LIMB(0x205a2eed73563032), CNST_LIMB(0xfd3663b27f31d529), CNST_LIMB(0x9ded549671832381), CNST_LIMB(0x9ef9f21eed31b7c1)},
        /* 242 */ {8, CNST_LIMB(0x2053ef71773d7e6a), CNST_LIMB(0xfd6753e032ea0efe), CNST_LIMB(0xa33f092e0b1ac100), CNST_LIMB(0x91747422be14b0b2)},
        /* 243 */ {8, CNST_LIMB(0x204db8f388552ea9), CNST_LIMB(0xfd9810643d6614c3), CNST_LIMB(0xa8b8b452291fe821), CNST_LIMB(0x846d550e37b5063d)},
        /* 244 */ {8, CNST_LIMB(0x20478b5cdbe2bb2f), CNST_LIMB(0xfdc899ab3ff56c5e), CNST_LIMB(0xae5b564ac3a10000), CNST_LIMB(0x77df79e9a96c06f6)},
        /* 245 */ {8, CNST_LIMB(0x20416696f957cfbf), CNST_LIMB(0xfdf8f02086af2c4b), CNST_LIMB(0xb427f4b3be74c361), CNST_LIMB(0x6bc6019636c7d0c2)},
        /* 246 */ {8, CNST_LIMB(0x203b4a8bb8d356e7), CNST_LIMB(0xfe29142e0e01401f), CNST_LIMB(0xba1f9a938041e100), CNST_LIMB(0x601c4205aebd9e47)},
        /* 247 */ {8, CNST_LIMB(0x2035372541ab0f0d), CNST_LIMB(0xfe59063c8822ce56), CNST_LIMB(0xc0435871d1110f41), CNST_LIMB(0x54ddc59756f05016)},
        /* 248 */ {8, CNST_LIMB(0x202f2c4e08fd6dcc), CNST_LIMB(0xfe88c6b3626a72aa), CNST_LIMB(0xc694446f01000000), CNST_LIMB(0x4a0648979c838c18)},
        /* 249 */ {8, CNST_LIMB(0x202929f0d04b99e9), CNST_LIMB(0xfeb855f8ca88fb0d), CNST_LIMB(0xcd137a5b57ac3ec1), CNST_LIMB(0x3f91b6e0bb3a053d)},
        /* 250 */ {8, CNST_LIMB(0x20232ff8a41b45eb), CNST_LIMB(0xfee7b471b3a9507d), CNST_LIMB(0xd3c21bcecceda100), CNST_LIMB(0x357c299a88ea76a5)},
        /* 251 */ {8, CNST_LIMB(0x201d3e50daa036db), CNST_LIMB(0xff16e281db76303b), CNST_LIMB(0xdaa150410b788de1), CNST_LIMB(0x2bc1e517aecc56e3)},
        /* 252 */ {8, CNST_LIMB(0x201754e5126d446d), CNST_LIMB(0xff45e08bcf06554e), CNST_LIMB(0xe1b24521be010000), CNST_LIMB(0x225f56ceb3da9f5d)},
        /* 253 */ {8, CNST_LIMB(0x201173a1312ca135), CNST_LIMB(0xff74aef0efafadd7), CNST_LIMB(0xe8f62df12777c1a1), CNST_LIMB(0x1951136d53ad63ac)},
        /* 254 */ {8, CNST_LIMB(0x200b9a71625f3b13), CNST_LIMB(0xffa34e1177c23362), CNST_LIMB(0xf06e445906fc0100), CNST_LIMB(0x1093d504b3cd7d93)},
        /* 255 */ {8, CNST_LIMB(0x2005c94216230568), CNST_LIMB(0xffd1be4c7f2af942), CNST_LIMB(0xf81bc845c81bf801), CNST_LIMB(0x824794d1ec1814f)},
        /* 256 */ {8, CNST_LIMB(0x1fffffffffffffff), CNST_LIMB(0xffffffffffffffff), CNST_LIMB(0x8), CNST_LIMB(0x0)},
};

const mp_limb_t
    __gmp_fib_table[FIB_TABLE_LIMIT + 2] = {
        CNST_LIMB(0x1),                /* -1 */
        CNST_LIMB(0x0),                /* 0 */
        CNST_LIMB(0x1),                /* 1 */
        CNST_LIMB(0x1),                /* 2 */
        CNST_LIMB(0x2),                /* 3 */
        CNST_LIMB(0x3),                /* 4 */
        CNST_LIMB(0x5),                /* 5 */
        CNST_LIMB(0x8),                /* 6 */
        CNST_LIMB(0xd),                /* 7 */
        CNST_LIMB(0x15),               /* 8 */
        CNST_LIMB(0x22),               /* 9 */
        CNST_LIMB(0x37),               /* 10 */
        CNST_LIMB(0x59),               /* 11 */
        CNST_LIMB(0x90),               /* 12 */
        CNST_LIMB(0xe9),               /* 13 */
        CNST_LIMB(0x179),              /* 14 */
        CNST_LIMB(0x262),              /* 15 */
        CNST_LIMB(0x3db),              /* 16 */
        CNST_LIMB(0x63d),              /* 17 */
        CNST_LIMB(0xa18),              /* 18 */
        CNST_LIMB(0x1055),             /* 19 */
        CNST_LIMB(0x1a6d),             /* 20 */
        CNST_LIMB(0x2ac2),             /* 21 */
        CNST_LIMB(0x452f),             /* 22 */
        CNST_LIMB(0x6ff1),             /* 23 */
        CNST_LIMB(0xb520),             /* 24 */
        CNST_LIMB(0x12511),            /* 25 */
        CNST_LIMB(0x1da31),            /* 26 */
        CNST_LIMB(0x2ff42),            /* 27 */
        CNST_LIMB(0x4d973),            /* 28 */
        CNST_LIMB(0x7d8b5),            /* 29 */
        CNST_LIMB(0xcb228),            /* 30 */
        CNST_LIMB(0x148add),           /* 31 */
        CNST_LIMB(0x213d05),           /* 32 */
        CNST_LIMB(0x35c7e2),           /* 33 */
        CNST_LIMB(0x5704e7),           /* 34 */
        CNST_LIMB(0x8cccc9),           /* 35 */
        CNST_LIMB(0xe3d1b0),           /* 36 */
        CNST_LIMB(0x1709e79),          /* 37 */
        CNST_LIMB(0x2547029),          /* 38 */
        CNST_LIMB(0x3c50ea2),          /* 39 */
        CNST_LIMB(0x6197ecb),          /* 40 */
        CNST_LIMB(0x9de8d6d),          /* 41 */
        CNST_LIMB(0xff80c38),          /* 42 */
        CNST_LIMB(0x19d699a5),         /* 43 */
        CNST_LIMB(0x29cea5dd),         /* 44 */
        CNST_LIMB(0x43a53f82),         /* 45 */
        CNST_LIMB(0x6d73e55f),         /* 46 */
        CNST_LIMB(0xb11924e1),         /* 47 */
        CNST_LIMB(0x11e8d0a40),        /* 48 */
        CNST_LIMB(0x1cfa62f21),        /* 49 */
        CNST_LIMB(0x2ee333961),        /* 50 */
        CNST_LIMB(0x4bdd96882),        /* 51 */
        CNST_LIMB(0x7ac0ca1e3),        /* 52 */
        CNST_LIMB(0xc69e60a65),        /* 53 */
        CNST_LIMB(0x1415f2ac48),       /* 54 */
        CNST_LIMB(0x207fd8b6ad),       /* 55 */
        CNST_LIMB(0x3495cb62f5),       /* 56 */
        CNST_LIMB(0x5515a419a2),       /* 57 */
        CNST_LIMB(0x89ab6f7c97),       /* 58 */
        CNST_LIMB(0xdec1139639),       /* 59 */
        CNST_LIMB(0x1686c8312d0),      /* 60 */
        CNST_LIMB(0x2472d96a909),      /* 61 */
        CNST_LIMB(0x3af9a19bbd9),      /* 62 */
        CNST_LIMB(0x5f6c7b064e2),      /* 63 */
        CNST_LIMB(0x9a661ca20bb),      /* 64 */
        CNST_LIMB(0xf9d297a859d),      /* 65 */
        CNST_LIMB(0x19438b44a658),     /* 66 */
        CNST_LIMB(0x28e0b4bf2bf5),     /* 67 */
        CNST_LIMB(0x42244003d24d),     /* 68 */
        CNST_LIMB(0x6b04f4c2fe42),     /* 69 */
        CNST_LIMB(0xad2934c6d08f),     /* 70 */
        CNST_LIMB(0x1182e2989ced1),    /* 71 */
        CNST_LIMB(0x1c5575e509f60),    /* 72 */
        CNST_LIMB(0x2dd8587da6e31),    /* 73 */
        CNST_LIMB(0x4a2dce62b0d91),    /* 74 */
        CNST_LIMB(0x780626e057bc2),    /* 75 */
        CNST_LIMB(0xc233f54308953),    /* 76 */
        CNST_LIMB(0x13a3a1c2360515),   /* 77 */
        CNST_LIMB(0x1fc6e116668e68),   /* 78 */
        CNST_LIMB(0x336a82d89c937d),   /* 79 */
        CNST_LIMB(0x533163ef0321e5),   /* 80 */
        CNST_LIMB(0x869be6c79fb562),   /* 81 */
        CNST_LIMB(0xd9cd4ab6a2d747),   /* 82 */
        CNST_LIMB(0x16069317e428ca9),  /* 83 */
        CNST_LIMB(0x23a367c34e563f0),  /* 84 */
        CNST_LIMB(0x39a9fadb327f099),  /* 85 */
        CNST_LIMB(0x5d4d629e80d5489),  /* 86 */
        CNST_LIMB(0x96f75d79b354522),  /* 87 */
        CNST_LIMB(0xf444c01834299ab),  /* 88 */
        CNST_LIMB(0x18b3c1d91e77decd), /* 89 */
        CNST_LIMB(0x27f80ddaa1ba7878), /* 90 */
        CNST_LIMB(0x40abcfb3c0325745), /* 91 */
        CNST_LIMB(0x68a3dd8e61eccfbd), /* 92 */
        CNST_LIMB(0xa94fad42221f2702), /* 93 */
};
#endif

        /* Stuff used by mpn/generic/perfsqr.c and mpz/prime_p.c */
#if GMP_NUMB_BITS == 2
#define PP 0x3 /* 3 */
#define PP_FIRST_OMITTED 5
#endif
#if GMP_NUMB_BITS == 4
#define PP 0xF /* 3 x 5 */
#define PP_FIRST_OMITTED 7
#endif
#if GMP_NUMB_BITS == 8
#define PP 0x69 /* 3 x 5 x 7 */
#define PP_FIRST_OMITTED 11
#endif
#if GMP_NUMB_BITS == 16
#define PP 0x3AA7 /* 3 x 5 x 7 x 11 x 13 */
#define PP_FIRST_OMITTED 17
#endif
#if GMP_NUMB_BITS == 32
#define PP 0xC0CFD797L /* 3 x 5 x 7 x 11 x ... x 29 */
#define PP_INVERTED 0x53E5645CL
#define PP_FIRST_OMITTED 31
#endif
#if GMP_NUMB_BITS == 64
#define PP CNST_LIMB(0xE221F97C30E94E1D) /* 3 x 5 x 7 x 11 x ... x 53 */
#define PP_INVERTED CNST_LIMB(0x21CFE6CFC938B36B)
#define PP_FIRST_OMITTED 59
#endif
#ifndef PP_FIRST_OMITTED
#define PP_FIRST_OMITTED 3
#endif

        typedef struct
        {
            mp_limb_t d0, d1;
        } mp_double_limb_t;
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_gcd_22 __GPGMP_MPN(gcd_22)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_double_limb_t gpmpn_gcd_22(mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t);
GPGMP_MPN_NAMESPACE_END

        /* BIT1 means a result value in bit 1 (second least significant bit), with a
           zero bit representing +1 and a one bit representing -1.  Bits other than
           bit 1 are garbage.  These are meant to be kept in "int"s, and casts are
           used to ensure the expressions are "int"s even if a and/or b might be
           other types.

           JACOBI_TWOS_U_BIT1 and JACOBI_RECIP_UU_BIT1 are used in gpmpn_jacobi_base
           and their speed is important.  Expressions are used rather than
           conditionals to accumulate sign changes, which effectively means XORs
           instead of conditional JUMPs. */

        /* (a/0), with a signed; is 1 if a=+/-1, 0 otherwise */
#define JACOBI_S0(a) (((a) == 1) | ((a) == -1))

        /* (a/0), with a unsigned; is 1 if a=+/-1, 0 otherwise */
#define JACOBI_U0(a) ((a) == 1)

        /* FIXME: JACOBI_LS0 and JACOBI_0LS are the same, so delete one and
           come up with a better name. */

        /* (a/0), with a given by low and size;
           is 1 if a=+/-1, 0 otherwise */
#define JACOBI_LS0(alow, asize) \
        (((asize) == 1 || (asize) == -1) && (alow) == 1)

        /* (a/0), with a an mpz_t;
           fetch of low limb always valid, even if size is zero */
#define JACOBI_Z0(a) JACOBI_LS0(PTR(a)[0], SIZ(a))

        /* (0/b), with b unsigned; is 1 if b=1, 0 otherwise */
#define JACOBI_0U(b) ((b) == 1)

        /* (0/b), with b unsigned; is 1 if b=+/-1, 0 otherwise */
#define JACOBI_0S(b) ((b) == 1 || (b) == -1)

        /* (0/b), with b given by low and size; is 1 if b=+/-1, 0 otherwise */
#define JACOBI_0LS(blow, bsize) \
        (((bsize) == 1 || (bsize) == -1) && (blow) == 1)

        /* Convert a bit1 to +1 or -1. */
#define JACOBI_BIT1_TO_PN(result_bit1) \
        (1 - ((int)(result_bit1) & 2))

        /* (2/b), with b unsigned and odd;
           is (-1)^((b^2-1)/8) which is 1 if b==1,7mod8 or -1 if b==3,5mod8 and
           hence obtained from (b>>1)^b */
#define JACOBI_TWO_U_BIT1(b) \
        ((int)(((b) >> 1) ^ (b)))

        /* (2/b)^twos, with b unsigned and odd */
#define JACOBI_TWOS_U_BIT1(twos, b) \
        ((int)((twos) << 1) & JACOBI_TWO_U_BIT1(b))

        /* (2/b)^twos, with b unsigned and odd */
#define JACOBI_TWOS_U(twos, b) \
        (JACOBI_BIT1_TO_PN(JACOBI_TWOS_U_BIT1(twos, b)))

        /* (-1/b), with b odd (signed or unsigned);
           is (-1)^((b-1)/2) */
#define JACOBI_N1B_BIT1(b) \
        ((int)(b))

        /* (a/b) effect due to sign of a: signed/unsigned, b odd;
           is (-1/b) if a<0, or +1 if a>=0 */
#define JACOBI_ASGN_SU_BIT1(a, b) \
        ((((a) < 0) << 1) & JACOBI_N1B_BIT1(b))

        /* (a/b) effect due to sign of b: signed/signed;
           is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_SS_BIT1(a, b) \
        ((((a) < 0) & ((b) < 0)) << 1)

        /* (a/b) effect due to sign of b: signed/mpz;
           is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_SZ_BIT1(a, b) \
        JACOBI_BSGN_SS_BIT1(a, SIZ(b))

        /* (a/b) effect due to sign of b: mpz/signed;
           is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_ZS_BIT1(a, b) \
        JACOBI_BSGN_SZ_BIT1(b, a)

        /* (a/b) reciprocity to switch to (b/a), a,b both unsigned and odd;
           is (-1)^((a-1)*(b-1)/4), which means +1 if either a,b==1mod4, or -1 if
           both a,b==3mod4, achieved in bit 1 by a&b.  No ASSERT()s about a,b odd
           because this is used in a couple of places with only bit 1 of a or b
           valid. */
#define JACOBI_RECIP_UU_BIT1(a, b) \
        ((int)((a) & (b)))

        /* Strip low zero limbs from {b_ptr,b_size} by incrementing b_ptr and
           decrementing b_size.  b_low should be b_ptr[0] on entry, and will be
           updated for the new b_ptr.  result_bit1 is updated according to the
           factors of 2 stripped, as per (a/2).  */
#define JACOBI_STRIP_LOW_ZEROS(result_bit1, a, b_ptr, b_size, b_low) \
        do                                                               \
        {                                                                \
            ASSERT((b_size) >= 1);                                       \
            ASSERT((b_low) == (b_ptr)[0]);                               \
                                                                         \
            while (UNLIKELY((b_low) == 0))                               \
            {                                                            \
                (b_size)--;                                              \
                ASSERT((b_size) >= 1);                                   \
                (b_ptr)++;                                               \
                (b_low) = *(b_ptr);                                      \
                                                                         \
                ASSERT(((a) & 1) != 0);                                  \
                if ((GMP_NUMB_BITS % 2) == 1)                            \
                    (result_bit1) ^= JACOBI_TWO_U_BIT1(a);               \
            }                                                            \
        } while (0)

        /* Set a_rem to {a_ptr,a_size} reduced modulo b, either using mod_1 or
           modexact_1_odd, but in either case leaving a_rem<b.  b must be odd and
           unsigned.  modexact_1_odd effectively calculates -a mod b, and
           result_bit1 is adjusted for the factor of -1.

           The way gpmpn_modexact_1_odd sometimes bases its remainder on a_size and
           sometimes on a_size-1 means if GMP_NUMB_BITS is odd we can't know what
           factor to introduce into result_bit1, so for that case use gpmpn_mod_1
           unconditionally.

           FIXME: gpmpn_modexact_1_odd is more efficient, so some way to get it used
           for odd GMP_NUMB_BITS would be good.  Perhaps it could mung its result,
           or not skip a divide step, or something. */

#define JACOBI_MOD_OR_MODEXACT_1_ODD(result_bit1, a_rem, a_ptr, a_size, b)                    \
        do                                                                                        \
        {                                                                                         \
            mp_srcptr __a_ptr = (a_ptr);                                                          \
            mp_size_t __a_size = (a_size);                                                        \
            mp_limb_t __b = (b);                                                                  \
                                                                                                  \
            ASSERT(__a_size >= 1);                                                                \
            ASSERT(__b & 1);                                                                      \
                                                                                                  \
            if ((GMP_NUMB_BITS % 2) != 0 || ABOVE_THRESHOLD(__a_size, BMOD_1_TO_MOD_1_THRESHOLD)) \
            {                                                                                     \
                (a_rem) = gpmpn_mod_1(__a_ptr, __a_size, __b);                                      \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                (result_bit1) ^= JACOBI_N1B_BIT1(__b);                                            \
                (a_rem) = gpmpn_modexact_1_odd(__a_ptr, __a_size, __b);                             \
            }                                                                                     \
        } while (0)

        /* State for the Jacobi computation using Lehmer. */

#define jacobi_table __gmp_jacobi_table

#ifdef __CUDA_ARCH__
__device__
#endif
        __GPGMP_DECLSPEC const unsigned char jacobi_table[208] = {
            0,
            0,
            0,
            0,
            0,
            12,
            8,
            4,
            1,
            1,
            1,
            1,
            1,
            13,
            9,
            5,
            2,
            2,
            2,
            2,
            2,
            6,
            10,
            14,
            3,
            3,
            3,
            3,
            3,
            7,
            11,
            15,
            4,
            16,
            6,
            18,
            4,
            0,
            12,
            8,
            5,
            17,
            7,
            19,
            5,
            1,
            13,
            9,
            6,
            18,
            4,
            16,
            6,
            10,
            14,
            2,
            7,
            19,
            5,
            17,
            7,
            11,
            15,
            3,
            8,
            10,
            9,
            11,
            8,
            4,
            0,
            12,
            9,
            11,
            8,
            10,
            9,
            5,
            1,
            13,
            10,
            9,
            11,
            8,
            10,
            14,
            2,
            6,
            11,
            8,
            10,
            9,
            11,
            15,
            3,
            7,
            12,
            22,
            24,
            20,
            12,
            8,
            4,
            0,
            13,
            23,
            25,
            21,
            13,
            9,
            5,
            1,
            25,
            21,
            13,
            23,
            14,
            2,
            6,
            10,
            24,
            20,
            12,
            22,
            15,
            3,
            7,
            11,
            16,
            6,
            18,
            4,
            16,
            16,
            16,
            16,
            17,
            7,
            19,
            5,
            17,
            17,
            17,
            17,
            18,
            4,
            16,
            6,
            18,
            22,
            19,
            23,
            19,
            5,
            17,
            7,
            19,
            23,
            18,
            22,
            20,
            12,
            22,
            24,
            20,
            20,
            20,
            20,
            21,
            13,
            23,
            25,
            21,
            21,
            21,
            21,
            22,
            24,
            20,
            12,
            22,
            19,
            23,
            18,
            23,
            25,
            21,
            13,
            23,
            18,
            22,
            19,
            24,
            20,
            12,
            22,
            15,
            3,
            7,
            11,
            25,
            21,
            13,
            23,
            14,
            2,
            6,
            10,
        };


        /* Bit layout for the initial state. b must be odd.

              3  2  1 0
           +--+--+--+--+
           |a1|a0|b1| s|
           +--+--+--+--+

         */
        ANYCALLER static inline unsigned
        gpmpn_jacobi_init(unsigned a, unsigned b, unsigned s)
        {
            ASSERT(b & 1);
            ASSERT(s <= 1);
            return ((a & 3) << 2) + (b & 2) + s;
        }

        ANYCALLER static inline int
        gpmpn_jacobi_finish(unsigned bits)
        {
            /* (a, b) = (1,0) or (0,1) */
            ASSERT((bits & 14) == 0);

            return 1 - 2 * (bits & 1);
        }

        ANYCALLER static inline unsigned
        gpmpn_jacobi_update(unsigned bits, unsigned denominator, unsigned q)
        {
            /* FIXME: Could halve table size by not including the e bit in the
             * index, and instead xor when updating. Then the lookup would be
             * like
             *
             *   bits ^= table[((bits & 30) << 2) + (denominator << 2) + q];
             */

            ASSERT(bits < 26);
            ASSERT(denominator < 2);
            ASSERT(q < 4);

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
GPGMP_MPN_NAMESPACE_BEGIN
        /* Matrix multiplication */
#define gpmpn_matrix22_mul __GPGMP_MPN(matrix22_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_matrix22_mul(mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
#define gpmpn_matrix22_mul_itch __GPGMP_MPN(matrix22_mul_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_matrix22_mul_itch(mp_size_t, mp_size_t) ATTRIBUTE_CONST;
GPGMP_MPN_NAMESPACE_END
#ifndef MATRIX22_STRASSEN_THRESHOLD
#define MATRIX22_STRASSEN_THRESHOLD 30
#endif

        /* HGCD definitions */

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

#define MPN_EXTRACT_NUMB(count, xh, xl)                      \
        ((((xh) << ((count) - GMP_NAIL_BITS)) & GMP_NUMB_MASK) | \
         ((xl) >> (GMP_LIMB_BITS - (count))))

        /* The matrix non-negative M = (u, u'; v,v') keeps track of the
           reduction (a;b) = M (alpha; beta) where alpha, beta are smaller
           than a, b. The determinant must always be one, so that M has an
           inverse (v', -u'; -v, u). Elements always fit in GMP_NUMB_BITS - 1
           bits. */
        struct hgcd_matrix1
        {
            mp_limb_t u[2][2];
        };
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_hgcd2 __GPGMP_MPN(hgcd2)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_hgcd2(mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t, struct hgcd_matrix1 *);

#define gpmpn_hgcd_mul_matrix1_vector __GPGMP_MPN(hgcd_mul_matrix1_vector)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_mul_matrix1_vector(const struct hgcd_matrix1 *, mp_ptr, mp_srcptr, mp_ptr, mp_size_t);

#define gpmpn_matrix22_mul1_inverse_vector __GPGMP_MPN(matrix22_mul1_inverse_vector)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_matrix22_mul1_inverse_vector(const struct hgcd_matrix1 *, mp_ptr, mp_srcptr, mp_ptr, mp_size_t);

#define gpmpn_hgcd2_jacobi __GPGMP_MPN(hgcd2_jacobi)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_hgcd2_jacobi(mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t, struct hgcd_matrix1 *, unsigned *);
GPGMP_MPN_NAMESPACE_END
        struct hgcd_matrix
        {
            mp_size_t alloc; /* for sanity checking only */
            mp_size_t n;
            mp_ptr p[2][2];
        };

#define MPN_HGCD_MATRIX_INIT_ITCH(n) (4 * ((n + 1) / 2 + 1))
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_hgcd_matrix_init __GPGMP_MPN(hgcd_matrix_init)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_hgcd_matrix_init(struct hgcd_matrix *, mp_size_t, mp_ptr);

#define gpmpn_hgcd_matrix_update_q __GPGMP_MPN(hgcd_matrix_update_q)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_hgcd_matrix_update_q(struct hgcd_matrix *, mp_srcptr, mp_size_t, unsigned, mp_ptr);

#define gpmpn_hgcd_matrix_mul_1 __GPGMP_MPN(hgcd_matrix_mul_1)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_hgcd_matrix_mul_1(struct hgcd_matrix *, const struct hgcd_matrix1 *, mp_ptr);

#define gpmpn_hgcd_matrix_mul __GPGMP_MPN(hgcd_matrix_mul)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_hgcd_matrix_mul(struct hgcd_matrix *, const struct hgcd_matrix *, mp_ptr);

#define gpmpn_hgcd_matrix_adjust __GPGMP_MPN(hgcd_matrix_adjust)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_matrix_adjust(const struct hgcd_matrix *, mp_size_t, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

#define gpmpn_hgcd_step __GPGMP_MPN(hgcd_step)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_step(mp_size_t, mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

#define gpmpn_hgcd_reduce __GPGMP_MPN(hgcd_reduce)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_reduce(struct hgcd_matrix *, mp_ptr, mp_ptr, mp_size_t, mp_size_t, mp_ptr);

#define gpmpn_hgcd_reduce_itch __GPGMP_MPN(hgcd_reduce_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_reduce_itch(mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_hgcd_itch __GPGMP_MPN(hgcd_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_itch(mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_hgcd __GPGMP_MPN(hgcd)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd(mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

#define gpmpn_hgcd_appr_itch __GPGMP_MPN(hgcd_appr_itch)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_appr_itch(mp_size_t) ATTRIBUTE_CONST;

#define gpmpn_hgcd_appr __GPGMP_MPN(hgcd_appr)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_hgcd_appr(mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

#define gpmpn_hgcd_jacobi __GPGMP_MPN(hgcd_jacobi)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_hgcd_jacobi(mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, unsigned *, mp_ptr);
GPGMP_MPN_NAMESPACE_END
        typedef void gcd_subdiv_step_hook(void *, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, int);

        /* Needs storage for the quotient */
#define MPN_GCD_SUBDIV_STEP_ITCH(n) (n)
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_gcd_subdiv_step __GPGMP_MPN(gcd_subdiv_step)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_gcd_subdiv_step(mp_ptr, mp_ptr, mp_size_t, mp_size_t, gcd_subdiv_step_hook *, void *, mp_ptr);
GPGMP_MPN_NAMESPACE_END
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
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_gcdext_hook __GPGMP_MPN(gcdext_hook)
        gcd_subdiv_step_hook gpmpn_gcdext_hook;
GPGMP_MPN_NAMESPACE_END

#define MPN_GCDEXT_LEHMER_N_ITCH(n) (4 * (n) + 3)
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_gcdext_lehmer_n __GPGMP_MPN(gcdext_lehmer_n)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_gcdext_lehmer_n(mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_ptr, mp_size_t, mp_ptr);
GPGMP_MPN_NAMESPACE_END
        /* 4*(an + 1) + 4*(bn + 1) + an */
#define MPN_GCDEXT_LEHMER_ITCH(an, bn) (5 * (an) + 4 * (bn) + 8)

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

        /* Definitions for gpmpn_set_str and gpmpn_get_str */
        struct powers
        {
            mp_ptr p;              /* actual power value */
            mp_size_t n;           /* # of limbs at p */
            mp_size_t shift;       /* weight of lowest limb, in limb base B */
            size_t digits_in_base; /* number of corresponding digits */
            int base;
        };
        typedef struct powers powers_t;
#define gpmpn_str_powtab_alloc(n) ((n) + 2 * GMP_LIMB_BITS) /* FIXME: This can perhaps be trimmed */
#define gpmpn_dc_set_str_itch(n) ((n) + GMP_LIMB_BITS)
#define gpmpn_dc_get_str_itch(n) ((n) + GMP_LIMB_BITS)
GPGMP_MPN_NAMESPACE_BEGIN
#define gpmpn_compute_powtab __GPGMP_MPN(compute_powtab)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpn_compute_powtab(powers_t *, mp_ptr, mp_size_t, int);
#define gpmpn_dc_set_str __GPGMP_MPN(dc_set_str)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_dc_set_str(mp_ptr, const unsigned char *, size_t, const powers_t *, mp_ptr);
#define gpmpn_bc_set_str __GPGMP_MPN(bc_set_str)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_bc_set_str(mp_ptr, const unsigned char *, size_t, int);
GPGMP_MPN_NAMESPACE_END
        /* __GMPF_BITS_TO_PREC applies a minimum 53 bits, rounds upwards to a whole
           limb and adds an extra limb.  __GMPF_PREC_TO_BITS drops that extra limb,
           hence giving back the user's size in bits rounded up.  Notice that
           converting prec->bits->prec gives an unchanged value.  */
#define __GMPF_BITS_TO_PREC(n) \
        ((mp_size_t)((__GMP_MAX(53, n) + 2 * GMP_NUMB_BITS - 1) / GMP_NUMB_BITS))
#define __GMPF_PREC_TO_BITS(n) \
        ((mp_bitcnt_t)(n) * GMP_NUMB_BITS - GMP_NUMB_BITS)

        __GPGMP_DECLSPEC extern mp_size_t __gmp_default_fp_limb_precision; // TODO: Don't overlook this when implementing MPF's

        /* Compute the number of base-b digits corresponding to nlimbs limbs, rounding
           down.  */
#define DIGITS_IN_BASE_PER_LIMB(res, nlimbs, b)                           \
        do                                                                    \
        {                                                                     \
            mp_limb_t _ph, _dummy;                                            \
            umul_ppmm(_ph, _dummy,                                            \
                      mp_bases[b].logb2, GMP_NUMB_BITS *(mp_limb_t)(nlimbs)); \
            res = _ph;                                                        \
        } while (0)

        /* Compute the number of limbs corresponding to ndigits base-b digits, rounding
           up.  */
#define LIMBS_PER_DIGIT_IN_BASE(res, ndigits, b)                         \
        do                                                                   \
        {                                                                    \
            mp_limb_t _ph, _dummy;                                           \
            umul_ppmm(_ph, _dummy, mp_bases[b].log2b, (mp_limb_t)(ndigits)); \
            res = 8 * _ph / GMP_NUMB_BITS + 2;                               \
        } while (0)

        /* Set n to the number of significant digits an mpf of the given _mp_prec
           field, in the given base.  This is a rounded up value, designed to ensure
           there's enough digits to reproduce all the guaranteed part of the value.

           There are prec many limbs, but the high might be only "1" so forget it
           and just count prec-1 limbs into chars.  +1 rounds that upwards, and a
           further +1 is because the limbs usually won't fall on digit boundaries.

           FIXME: If base is a power of 2 and the bits per digit divides
           GMP_LIMB_BITS then the +2 is unnecessary.  This happens always for
           base==2, and in base==16 with the current 32 or 64 bit limb sizes. */

#define MPF_SIGNIFICANT_DIGITS(n, base, prec)            \
        do                                                   \
        {                                                    \
            size_t rawn;                                     \
            ASSERT(base >= 2 && base < numberof(mp_bases));  \
            DIGITS_IN_BASE_PER_LIMB(rawn, (prec) - 1, base); \
            n = rawn + 2;                                    \
        } while (0)

        /* Decimal point string, from the current C locale.  Needs <langinfo.h> for
           nl_langinfo and constants, preferably with _GNU_SOURCE defined to get
           DECIMAL_POINT from glibc, and needs <locale.h> for localeconv, each under
           their respective #if HAVE_FOO_H.

           GLIBC recommends nl_langinfo because getting only one facet can be
           faster, apparently. */

        /* DECIMAL_POINT seems to need _GNU_SOURCE defined to get it from glibc. */
#if HAVE_NL_LANGINFO && defined(DECIMAL_POINT)
#define GMP_DECIMAL_POINT (nl_langinfo(DECIMAL_POINT))
#endif
        /* RADIXCHAR is deprecated, still in unix98 or some such. */
#if HAVE_NL_LANGINFO && defined(RADIXCHAR) && !defined(GMP_DECIMAL_POINT)
#define GMP_DECIMAL_POINT (nl_langinfo(RADIXCHAR))
#endif
        /* localeconv is slower since it returns all locale stuff */
#if HAVE_LOCALECONV && !defined(GMP_DECIMAL_POINT)
#define GMP_DECIMAL_POINT (localeconv()->decimal_point)
#endif
#if !defined(GMP_DECIMAL_POINT)
#define GMP_DECIMAL_POINT (".")
#endif

#define DOPRNT_CONV_FIXED 1
#define DOPRNT_CONV_SCIENTIFIC 2
#define DOPRNT_CONV_GENERAL 3

#define DOPRNT_JUSTIFY_NONE 0
#define DOPRNT_JUSTIFY_LEFT 1
#define DOPRNT_JUSTIFY_RIGHT 2
#define DOPRNT_JUSTIFY_INTERNAL 3

#define DOPRNT_SHOWBASE_YES 1
#define DOPRNT_SHOWBASE_NO 2
#define DOPRNT_SHOWBASE_NONZERO 3

        struct doprnt_params_t
        {
            int base;           /* negative for upper case */
            int conv;           /* choices above */
            const char *expfmt; /* exponent format */
            int exptimes4;      /* exponent multiply by 4 */
            char fill;          /* character */
            int justify;        /* choices above */
            int prec;           /* prec field, or -1 for all digits */
            int showbase;       /* choices above */
            int showpoint;      /* if radix point always shown */
            int showtrailing;   /* if trailing zeros wanted */
            char sign;          /* '+', ' ', or '\0' */
            int width;          /* width field */
        };

#if _GMP_H_HAVE_VA_LIST

        typedef int (*doprnt_format_t)(void *, const char *, va_list);
        typedef int (*doprnt_memory_t)(void *, const char *, size_t);
        typedef int (*doprnt_reps_t)(void *, int, int);
        typedef int (*doprnt_final_t)(void *);

        struct doprnt_funs_t
        {
            doprnt_format_t format;
            doprnt_memory_t memory;
            doprnt_reps_t reps;
            doprnt_final_t final; /* NULL if not required */
        };

        extern const struct doprnt_funs_t __gmp_fprintf_funs;
        extern const struct doprnt_funs_t __gmp_sprintf_funs;
        extern const struct doprnt_funs_t __gmp_snprintf_funs;
        extern const struct doprnt_funs_t __gmp_obstack_printf_funs;
        extern const struct doprnt_funs_t __gmp_ostream_funs;

        /* "buf" is a __gpgmp_allocate_func block of "alloc" many bytes.  The first
           "size" of these have been written.  "alloc > size" is maintained, so
           there's room to store a '\0' at the end.  "result" is where the
           application wants the final block pointer.  */
        struct gmp_asprintf_t
        {
            char **result;
            char *buf;
            size_t size;
            size_t alloc;
        };

#define GMP_ASPRINTF_T_INIT(d, output)                       \
        do                                                       \
        {                                                        \
            (d).result = (output);                               \
            (d).alloc = 256;                                     \
            (d).buf = (char *)(*__gpgmp_allocate_func)((d).alloc); \
            (d).size = 0;                                        \
        } while (0)

        /* If a realloc is necessary, use twice the size actually required, so as to
           avoid repeated small reallocs.  */
#define GMP_ASPRINTF_T_NEED(d, n)                                         \
        do                                                                    \
        {                                                                     \
            size_t alloc, newsize, newalloc;                                  \
            ASSERT((d)->alloc >= (d)->size + 1);                              \
                                                                              \
            alloc = (d)->alloc;                                               \
            newsize = (d)->size + (n);                                        \
            if (alloc <= newsize)                                             \
            {                                                                 \
                newalloc = 2 * newsize;                                       \
                (d)->alloc = newalloc;                                        \
                (d)->buf = __GMP_REALLOCATE_FUNC_TYPE((d)->buf,               \
                                                      alloc, newalloc, char); \
            }                                                                 \
        } while (0)
GPGMP_MPN_NAMESPACE_BEGIN
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_asprintf_memory(struct gmp_asprintf_t *, const char *, size_t);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_asprintf_reps(struct gmp_asprintf_t *, int, int);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_asprintf_final(struct gmp_asprintf_t *);
GPGMP_MPN_NAMESPACE_END
        /* buf is where to write the next output, and size is how much space is left
           there.  If the application passed size==0 then that's what we'll have
           here, and nothing at all should be written.  */
        struct gmp_snprintf_t
        {
            char *buf;
            size_t size;
        };

        /* Add the bytes printed by the call to the total retval, or bail out on an
           error.  */
#define DOPRNT_ACCUMULATE(call) \
        do                          \
        {                           \
            int __ret;              \
            __ret = call;           \
            if (__ret == -1)        \
                goto error;         \
            retval += __ret;        \
        } while (0)
#define DOPRNT_ACCUMULATE_FUN(fun, params) \
        do                                     \
        {                                      \
            ASSERT((fun) != NULL);             \
            DOPRNT_ACCUMULATE((*(fun))params); \
        } while (0)

#define DOPRNT_FORMAT(fmt, ap) \
        DOPRNT_ACCUMULATE_FUN(funs->format, (data, fmt, ap))
#define DOPRNT_MEMORY(ptr, len) \
        DOPRNT_ACCUMULATE_FUN(funs->memory, (data, ptr, len))
#define DOPRNT_REPS(c, n) \
        DOPRNT_ACCUMULATE_FUN(funs->reps, (data, c, n))

#define DOPRNT_STRING(str) DOPRNT_MEMORY(str, strlen(str))

#define DOPRNT_REPS_MAYBE(c, n) \
        do                          \
        {                           \
            if ((n) != 0)           \
                DOPRNT_REPS(c, n);  \
        } while (0)
#define DOPRNT_MEMORY_MAYBE(ptr, len) \
        do                                \
        {                                 \
            if ((len) != 0)               \
                DOPRNT_MEMORY(ptr, len);  \
        } while (0)
GPGMP_MPN_NAMESPACE_BEGIN
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_doprnt(const struct doprnt_funs_t *, void *, const char *, va_list);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_doprnt_integer(const struct doprnt_funs_t *, void *, const struct doprnt_params_t *, const char *);

#define __gmp_doprnt_mpf __gmp_doprnt_mpf2
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_doprnt_mpf(const struct doprnt_funs_t *, void *, const struct doprnt_params_t *, const char *, mpf_srcptr);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_replacement_vsnprintf(char *, size_t, const char *, va_list);
GPGMP_MPN_NAMESPACE_END
#endif /* _GMP_H_HAVE_VA_LIST */

        typedef int (*gmp_doscan_scan_t)(void *, const char *, ...);
        typedef void *(*gmp_doscan_step_t)(void *, int);
        typedef int (*gmp_doscan_get_t)(void *);
        typedef int (*gmp_doscan_unget_t)(int, void *);

        struct gmp_doscan_funs_t
        {
            gmp_doscan_scan_t scan;
            gmp_doscan_step_t step;
            gmp_doscan_get_t get;
            gmp_doscan_unget_t unget;
        };
        extern const struct gmp_doscan_funs_t __gmp_fscanf_funs;
        extern const struct gmp_doscan_funs_t __gmp_sscanf_funs;

#if _GMP_H_HAVE_VA_LIST
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int __gmp_doscan(const struct gmp_doscan_funs_t *, void *, const char *, va_list);
#endif

        /* For testing and debugging.  */
#define MPZ_CHECK_FORMAT(z)                                      \
        do                                                           \
        {                                                            \
            ASSERT_ALWAYS(SIZ(z) == 0 || PTR(z)[ABSIZ(z) - 1] != 0); \
            ASSERT_ALWAYS(ALLOC(z) >= ABSIZ(z));                     \
            ASSERT_ALWAYS_MPN(PTR(z), ABSIZ(z));                     \
        } while (0)

#define MPQ_CHECK_FORMAT(q)                                                       \
        do                                                                            \
        {                                                                             \
            MPZ_CHECK_FORMAT(mpq_numref(q));                                          \
            MPZ_CHECK_FORMAT(mpq_denref(q));                                          \
            ASSERT_ALWAYS(SIZ(mpq_denref(q)) >= 1);                                   \
                                                                                      \
            if (SIZ(mpq_numref(q)) == 0)                                              \
            {                                                                         \
                /* should have zero as 0/1 */                                         \
                ASSERT_ALWAYS(SIZ(mpq_denref(q)) == 1 && PTR(mpq_denref(q))[0] == 1); \
            }                                                                         \
            else                                                                      \
            {                                                                         \
                /* should have no common factors */                                   \
                mpz_t g;                                                              \
                mpz_init(g);                                                          \
                mpz_gcd(g, mpq_numref(q), mpq_denref(q));                             \
                ASSERT_ALWAYS(mpz_cmp_ui(g, 1) == 0);                                 \
                mpz_clear(g);                                                         \
            }                                                                         \
        } while (0)

#define MPF_CHECK_FORMAT(f)                                \
        do                                                     \
        {                                                      \
            ASSERT_ALWAYS(PREC(f) >= __GMPF_BITS_TO_PREC(53)); \
            ASSERT_ALWAYS(ABSIZ(f) <= PREC(f) + 1);            \
            if (SIZ(f) == 0)                                   \
                ASSERT_ALWAYS(EXP(f) == 0);                    \
            if (SIZ(f) != 0)                                   \
                ASSERT_ALWAYS(PTR(f)[ABSIZ(f) - 1] != 0);      \
        } while (0)

        /* Enhancement: The "mod" and "gcd_1" functions below could have
           __GMP_ATTRIBUTE_PURE, but currently (gcc 3.3) that's not supported on
           function pointers, only actual functions.  It probably doesn't make much
           difference to the gmp code, since hopefully we arrange calls so there's
           no great need for the compiler to move things around.  */

#if WANT_FAT_BINARY && (HAVE_HOST_CPU_FAMILY_x86 || HAVE_HOST_CPU_FAMILY_x86_64)
        /* NOTE: The function pointers in this struct are also in CPUVEC_FUNCS_LIST
           in mpn/x86/x86-defs.m4 and mpn/x86_64/x86_64-defs.m4.  Be sure to update
           those when changing here.  */
        struct cpuvec_t
        {
            DECL_add_n((*add_n));
            DECL_addlsh1_n((*addlsh1_n));
            DECL_addlsh2_n((*addlsh2_n));
            DECL_addmul_1((*addmul_1));
            DECL_addmul_2((*addmul_2));
            DECL_bdiv_dbm1c((*bdiv_dbm1c));
            DECL_cnd_add_n((*cnd_add_n));
            DECL_cnd_sub_n((*cnd_sub_n));
            DECL_com((*com));
            DECL_copyd((*copyd));
            DECL_copyi((*copyi));
            DECL_divexact_1((*divexact_1));
            DECL_divrem_1((*divrem_1));
            DECL_gcd_11((*gcd_11));
            DECL_lshift((*lshift));
            DECL_lshiftc((*lshiftc));
            DECL_mod_1((*mod_1));
            DECL_mod_1_1p((*mod_1_1p));
            DECL_mod_1_1p_cps((*mod_1_1p_cps));
            DECL_mod_1s_2p((*mod_1s_2p));
            DECL_mod_1s_2p_cps((*mod_1s_2p_cps));
            DECL_mod_1s_4p((*mod_1s_4p));
            DECL_mod_1s_4p_cps((*mod_1s_4p_cps));
            DECL_mod_34lsub1((*mod_34lsub1));
            DECL_modexact_1c_odd((*modexact_1c_odd));
            DECL_mul_1((*mul_1));
            DECL_mul_basecase((*mul_basecase));
            DECL_mullo_basecase((*mullo_basecase));
            DECL_preinv_divrem_1((*preinv_divrem_1));
            DECL_preinv_mod_1((*preinv_mod_1));
            DECL_redc_1((*redc_1));
            DECL_redc_2((*redc_2));
            DECL_rshift((*rshift));
            DECL_sqr_basecase((*sqr_basecase));
            DECL_sub_n((*sub_n));
            DECL_sublsh1_n((*sublsh1_n));
            DECL_submul_1((*submul_1));
            mp_size_t mul_toom22_threshold;
            mp_size_t mul_toom33_threshold;
            mp_size_t sqr_toom2_threshold;
            mp_size_t sqr_toom3_threshold;
            mp_size_t bmod_1_to_mod_1_threshold;
        };
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE extern struct cpuvec_t __ggpmpn_cpuvec;
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE extern int __ggpmpn_cpuvec_initialized;
#endif /* x86 fat binary */

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __ggpmpn_cpuvec_init(void);

        /* Get a threshold "field" from __ggpmpn_cpuvec, running __ggpmpn_cpuvec_init()
           if that hasn't yet been done (to establish the right values).  */
#define CPUVEC_THRESHOLD(field)                                           \
        ((LIKELY(__ggpmpn_cpuvec_initialized) ? 0 : (__ggpmpn_cpuvec_init(), 0)), \
         __ggpmpn_cpuvec.field)

GPGMP_MPN_NAMESPACE_BEGIN
#if HAVE_NATIVE_gpmpn_add_nc
#define gpmpn_add_nc __GPGMP_MPN(add_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#else

ANYCALLER static inline mp_limb_t
gpmpn_add_nc(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n, mp_limb_t ci)
{
    mp_limb_t co;
    co = gpmpn_add_n(rp, up, vp, n);
    co += gpmpn_add_1(rp, rp, n, ci);
    return co;
}
#endif

#if HAVE_NATIVE_gpmpn_sub_nc
#define gpmpn_sub_nc __GPGMP_MPN(sub_nc)
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_nc(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#else
ANYCALLER static inline mp_limb_t
gpmpn_sub_nc(mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n, mp_limb_t ci)
{
    mp_limb_t co;
    co = gpmpn_sub_n(rp, up, vp, n);
    co += gpmpn_sub_1(rp, rp, n, ci);
    return co;
}
#endif
GPGMP_MPN_NAMESPACE_END
#if TUNE_PROGRAM_BUILD
        /* Some extras wanted when recompiling some .c files for use by the tune
           program.  Not part of a normal build.

           It's necessary to keep these thresholds as #defines (just to an
           identically named variable), since various defaults are established based
           on #ifdef in the .c files.  For some this is not so (the defaults are
           instead established above), but all are done this way for consistency. */

#undef MUL_TOOM22_THRESHOLD
#define MUL_TOOM22_THRESHOLD mul_toom22_threshold
        extern mp_size_t mul_toom22_threshold;

#undef MUL_TOOM33_THRESHOLD
#define MUL_TOOM33_THRESHOLD mul_toom33_threshold
        extern mp_size_t mul_toom33_threshold;

#undef MUL_TOOM44_THRESHOLD
#define MUL_TOOM44_THRESHOLD mul_toom44_threshold
        extern mp_size_t mul_toom44_threshold;

#undef MUL_TOOM6H_THRESHOLD
#define MUL_TOOM6H_THRESHOLD mul_toom6h_threshold
        extern mp_size_t mul_toom6h_threshold;

#undef MUL_TOOM8H_THRESHOLD
#define MUL_TOOM8H_THRESHOLD mul_toom8h_threshold
        extern mp_size_t mul_toom8h_threshold;

#undef MUL_TOOM32_TO_TOOM43_THRESHOLD
#define MUL_TOOM32_TO_TOOM43_THRESHOLD mul_toom32_to_toom43_threshold
        extern mp_size_t mul_toom32_to_toom43_threshold;

#undef MUL_TOOM32_TO_TOOM53_THRESHOLD
#define MUL_TOOM32_TO_TOOM53_THRESHOLD mul_toom32_to_toom53_threshold
        extern mp_size_t mul_toom32_to_toom53_threshold;

#undef MUL_TOOM42_TO_TOOM53_THRESHOLD
#define MUL_TOOM42_TO_TOOM53_THRESHOLD mul_toom42_to_toom53_threshold
        extern mp_size_t mul_toom42_to_toom53_threshold;

#undef MUL_TOOM42_TO_TOOM63_THRESHOLD
#define MUL_TOOM42_TO_TOOM63_THRESHOLD mul_toom42_to_toom63_threshold
        extern mp_size_t mul_toom42_to_toom63_threshold;

#undef MUL_TOOM43_TO_TOOM54_THRESHOLD
#define MUL_TOOM43_TO_TOOM54_THRESHOLD mul_toom43_to_toom54_threshold;
        extern mp_size_t mul_toom43_to_toom54_threshold;

#undef MUL_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD mul_fft_threshold
        extern mp_size_t mul_fft_threshold;

#undef MUL_FFT_MODF_THRESHOLD
#define MUL_FFT_MODF_THRESHOLD mul_fft_modf_threshold
        extern mp_size_t mul_fft_modf_threshold;

#undef MUL_FFT_TABLE
#define MUL_FFT_TABLE {0}

#undef MUL_FFT_TABLE3
#define MUL_FFT_TABLE3 {{0, 0}}

        /* A native gpmpn_sqr_basecase is not tuned and SQR_BASECASE_THRESHOLD should
           remain as zero (always use it). */
#if !HAVE_NATIVE_gpmpn_sqr_basecase
#undef SQR_BASECASE_THRESHOLD
#define SQR_BASECASE_THRESHOLD sqr_basecase_threshold
        extern mp_size_t sqr_basecase_threshold;
#endif

#if TUNE_PROGRAM_BUILD_SQR
#undef SQR_TOOM2_THRESHOLD
#define SQR_TOOM2_THRESHOLD SQR_TOOM2_MAX_GENERIC
#else
#undef SQR_TOOM2_THRESHOLD
#define SQR_TOOM2_THRESHOLD sqr_toom2_threshold
    extern mp_size_t sqr_toom2_threshold;
#endif

#undef SQR_TOOM3_THRESHOLD
#define SQR_TOOM3_THRESHOLD sqr_toom3_threshold
        extern mp_size_t sqr_toom3_threshold;

#undef SQR_TOOM4_THRESHOLD
#define SQR_TOOM4_THRESHOLD sqr_toom4_threshold
        extern mp_size_t sqr_toom4_threshold;

#undef SQR_TOOM6_THRESHOLD
#define SQR_TOOM6_THRESHOLD sqr_toom6_threshold
        extern mp_size_t sqr_toom6_threshold;

#undef SQR_TOOM8_THRESHOLD
#define SQR_TOOM8_THRESHOLD sqr_toom8_threshold
        extern mp_size_t sqr_toom8_threshold;

#undef SQR_FFT_THRESHOLD
#define SQR_FFT_THRESHOLD sqr_fft_threshold
        extern mp_size_t sqr_fft_threshold;

#undef SQR_FFT_MODF_THRESHOLD
#define SQR_FFT_MODF_THRESHOLD sqr_fft_modf_threshold
        extern mp_size_t sqr_fft_modf_threshold;

#undef SQR_FFT_TABLE
#define SQR_FFT_TABLE {0}

#undef SQR_FFT_TABLE3
#define SQR_FFT_TABLE3 {{0, 0}}

#undef MULLO_BASECASE_THRESHOLD
#define MULLO_BASECASE_THRESHOLD mullo_basecase_threshold
        extern mp_size_t mullo_basecase_threshold;

#undef MULLO_DC_THRESHOLD
#define MULLO_DC_THRESHOLD mullo_dc_threshold
        extern mp_size_t mullo_dc_threshold;

#undef MULLO_MUL_N_THRESHOLD
#define MULLO_MUL_N_THRESHOLD mullo_mul_n_threshold
        extern mp_size_t mullo_mul_n_threshold;

#undef SQRLO_BASECASE_THRESHOLD
#define SQRLO_BASECASE_THRESHOLD sqrlo_basecase_threshold
        extern mp_size_t sqrlo_basecase_threshold;

#undef SQRLO_DC_THRESHOLD
#define SQRLO_DC_THRESHOLD sqrlo_dc_threshold
        extern mp_size_t sqrlo_dc_threshold;

#undef SQRLO_SQR_THRESHOLD
#define SQRLO_SQR_THRESHOLD sqrlo_sqr_threshold
        extern mp_size_t sqrlo_sqr_threshold;

#undef MULMID_TOOM42_THRESHOLD
#define MULMID_TOOM42_THRESHOLD mulmid_toom42_threshold
        extern mp_size_t mulmid_toom42_threshold;

#undef DIV_QR_2_PI2_THRESHOLD
#define DIV_QR_2_PI2_THRESHOLD div_qr_2_pi2_threshold
        extern mp_size_t div_qr_2_pi2_threshold;

#undef DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD dc_div_qr_threshold
        extern mp_size_t dc_div_qr_threshold;

#undef DC_DIVAPPR_Q_THRESHOLD
#define DC_DIVAPPR_Q_THRESHOLD dc_divappr_q_threshold
        extern mp_size_t dc_divappr_q_threshold;

#undef DC_BDIV_Q_THRESHOLD
#define DC_BDIV_Q_THRESHOLD dc_bdiv_q_threshold
        extern mp_size_t dc_bdiv_q_threshold;

#undef DC_BDIV_QR_THRESHOLD
#define DC_BDIV_QR_THRESHOLD dc_bdiv_qr_threshold
        extern mp_size_t dc_bdiv_qr_threshold;

#undef MU_DIV_QR_THRESHOLD
#define MU_DIV_QR_THRESHOLD mu_div_qr_threshold
        extern mp_size_t mu_div_qr_threshold;

#undef MU_DIVAPPR_Q_THRESHOLD
#define MU_DIVAPPR_Q_THRESHOLD mu_divappr_q_threshold
        extern mp_size_t mu_divappr_q_threshold;

#undef MUPI_DIV_QR_THRESHOLD
#define MUPI_DIV_QR_THRESHOLD mupi_div_qr_threshold
        extern mp_size_t mupi_div_qr_threshold;

#undef MU_BDIV_QR_THRESHOLD
#define MU_BDIV_QR_THRESHOLD mu_bdiv_qr_threshold
        extern mp_size_t mu_bdiv_qr_threshold;

#undef MU_BDIV_Q_THRESHOLD
#define MU_BDIV_Q_THRESHOLD mu_bdiv_q_threshold
        extern mp_size_t mu_bdiv_q_threshold;

#undef INV_MULMOD_BNM1_THRESHOLD
#define INV_MULMOD_BNM1_THRESHOLD inv_mulmod_bnm1_threshold
        extern mp_size_t inv_mulmod_bnm1_threshold;

#undef INV_NEWTON_THRESHOLD
#define INV_NEWTON_THRESHOLD inv_newton_threshold
        extern mp_size_t inv_newton_threshold;

#undef INV_APPR_THRESHOLD
#define INV_APPR_THRESHOLD inv_appr_threshold
        extern mp_size_t inv_appr_threshold;

#undef BINV_NEWTON_THRESHOLD
#define BINV_NEWTON_THRESHOLD binv_newton_threshold
        extern mp_size_t binv_newton_threshold;

#undef REDC_1_TO_REDC_2_THRESHOLD
#define REDC_1_TO_REDC_2_THRESHOLD redc_1_to_redc_2_threshold
        extern mp_size_t redc_1_to_redc_2_threshold;

#undef REDC_2_TO_REDC_N_THRESHOLD
#define REDC_2_TO_REDC_N_THRESHOLD redc_2_to_redc_n_threshold
        extern mp_size_t redc_2_to_redc_n_threshold;

#undef REDC_1_TO_REDC_N_THRESHOLD
#define REDC_1_TO_REDC_N_THRESHOLD redc_1_to_redc_n_threshold
        extern mp_size_t redc_1_to_redc_n_threshold;

#undef MATRIX22_STRASSEN_THRESHOLD
#define MATRIX22_STRASSEN_THRESHOLD matrix22_strassen_threshold
        extern mp_size_t matrix22_strassen_threshold;

        typedef int hgcd2_func_t(mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t,
                                 struct hgcd_matrix1 *);
        extern hgcd2_func_t *hgcd2_func;

#undef HGCD_THRESHOLD
#define HGCD_THRESHOLD hgcd_threshold
        extern mp_size_t hgcd_threshold;

#undef HGCD_APPR_THRESHOLD
#define HGCD_APPR_THRESHOLD hgcd_appr_threshold
        extern mp_size_t hgcd_appr_threshold;

#undef HGCD_REDUCE_THRESHOLD
#define HGCD_REDUCE_THRESHOLD hgcd_reduce_threshold
        extern mp_size_t hgcd_reduce_threshold;

#undef GCD_DC_THRESHOLD
#define GCD_DC_THRESHOLD gcd_dc_threshold
        extern mp_size_t gcd_dc_threshold;

#undef GCDEXT_DC_THRESHOLD
#define GCDEXT_DC_THRESHOLD gcdext_dc_threshold
        extern mp_size_t gcdext_dc_threshold;

#undef DIV_QR_1N_PI1_METHOD
#define DIV_QR_1N_PI1_METHOD div_qr_1n_pi1_method
        extern int div_qr_1n_pi1_method;

#undef DIV_QR_1_NORM_THRESHOLD
#define DIV_QR_1_NORM_THRESHOLD div_qr_1_norm_threshold
        extern mp_size_t div_qr_1_norm_threshold;

#undef DIV_QR_1_UNNORM_THRESHOLD
#define DIV_QR_1_UNNORM_THRESHOLD div_qr_1_unnorm_threshold
        extern mp_size_t div_qr_1_unnorm_threshold;

#undef DIVREM_1_NORM_THRESHOLD
#define DIVREM_1_NORM_THRESHOLD divrem_1_norm_threshold
        extern mp_size_t divrem_1_norm_threshold;

#undef DIVREM_1_UNNORM_THRESHOLD
#define DIVREM_1_UNNORM_THRESHOLD divrem_1_unnorm_threshold
        extern mp_size_t divrem_1_unnorm_threshold;

#undef MOD_1_NORM_THRESHOLD
#define MOD_1_NORM_THRESHOLD mod_1_norm_threshold
        extern mp_size_t mod_1_norm_threshold;

#undef MOD_1_UNNORM_THRESHOLD
#define MOD_1_UNNORM_THRESHOLD mod_1_unnorm_threshold
        extern mp_size_t mod_1_unnorm_threshold;

#undef MOD_1_1P_METHOD
#define MOD_1_1P_METHOD mod_1_1p_method
        extern int mod_1_1p_method;

#undef MOD_1N_TO_MOD_1_1_THRESHOLD
#define MOD_1N_TO_MOD_1_1_THRESHOLD mod_1n_to_mod_1_1_threshold
        extern mp_size_t mod_1n_to_mod_1_1_threshold;

#undef MOD_1U_TO_MOD_1_1_THRESHOLD
#define MOD_1U_TO_MOD_1_1_THRESHOLD mod_1u_to_mod_1_1_threshold
        extern mp_size_t mod_1u_to_mod_1_1_threshold;

#undef MOD_1_1_TO_MOD_1_2_THRESHOLD
#define MOD_1_1_TO_MOD_1_2_THRESHOLD mod_1_1_to_mod_1_2_threshold
        extern mp_size_t mod_1_1_to_mod_1_2_threshold;

#undef MOD_1_2_TO_MOD_1_4_THRESHOLD
#define MOD_1_2_TO_MOD_1_4_THRESHOLD mod_1_2_to_mod_1_4_threshold
        extern mp_size_t mod_1_2_to_mod_1_4_threshold;

#undef PREINV_MOD_1_TO_MOD_1_THRESHOLD
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD preinv_mod_1_to_mod_1_threshold
        extern mp_size_t preinv_mod_1_to_mod_1_threshold;

#if !UDIV_PREINV_ALWAYS
#undef DIVREM_2_THRESHOLD
#define DIVREM_2_THRESHOLD divrem_2_threshold
        extern mp_size_t divrem_2_threshold;
#endif

#undef MULMOD_BNM1_THRESHOLD
#define MULMOD_BNM1_THRESHOLD mulmod_bnm1_threshold
        extern mp_size_t mulmod_bnm1_threshold;

#undef SQRMOD_BNM1_THRESHOLD
#define SQRMOD_BNM1_THRESHOLD sqrmod_bnm1_threshold
        extern mp_size_t sqrmod_bnm1_threshold;

#undef GET_STR_DC_THRESHOLD
#define GET_STR_DC_THRESHOLD get_str_dc_threshold
        extern mp_size_t get_str_dc_threshold;

#undef GET_STR_PRECOMPUTE_THRESHOLD
#define GET_STR_PRECOMPUTE_THRESHOLD get_str_precompute_threshold
        extern mp_size_t get_str_precompute_threshold;

#undef SET_STR_DC_THRESHOLD
#define SET_STR_DC_THRESHOLD set_str_dc_threshold
        extern mp_size_t set_str_dc_threshold;

#undef SET_STR_PRECOMPUTE_THRESHOLD
#define SET_STR_PRECOMPUTE_THRESHOLD set_str_precompute_threshold
        extern mp_size_t set_str_precompute_threshold;

#undef FAC_ODD_THRESHOLD
#define FAC_ODD_THRESHOLD fac_odd_threshold
        extern mp_size_t fac_odd_threshold;

#undef FAC_DSC_THRESHOLD
#define FAC_DSC_THRESHOLD fac_dsc_threshold
        extern mp_size_t fac_dsc_threshold;

#undef FFT_TABLE_ATTRS
#define FFT_TABLE_ATTRS
        extern mp_size_t gpmpn_fft_table[2][MPN_FFT_TABLE_SIZE];
#define FFT_TABLE3_SIZE 2000 /* generous space for tuning */
        extern struct fft_table_nk gpmpn_fft_table3[2][FFT_TABLE3_SIZE];

        /* Sizes the tune program tests up to, used in a couple of recompilations. */
#undef MUL_TOOM22_THRESHOLD_LIMIT
#undef MUL_TOOM33_THRESHOLD_LIMIT
#undef MULLO_BASECASE_THRESHOLD_LIMIT
#undef SQRLO_BASECASE_THRESHOLD_LIMIT
#undef SQRLO_DC_THRESHOLD_LIMIT
#undef SQR_TOOM3_THRESHOLD_LIMIT
#define SQR_TOOM2_MAX_GENERIC 200
#define MUL_TOOM22_THRESHOLD_LIMIT 700
#define MUL_TOOM33_THRESHOLD_LIMIT 700
#define SQR_TOOM3_THRESHOLD_LIMIT 400
#define MUL_TOOM44_THRESHOLD_LIMIT 1000
#define SQR_TOOM4_THRESHOLD_LIMIT 1000
#define MUL_TOOM6H_THRESHOLD_LIMIT 1100
#define SQR_TOOM6_THRESHOLD_LIMIT 1100
#define MUL_TOOM8H_THRESHOLD_LIMIT 1200
#define SQR_TOOM8_THRESHOLD_LIMIT 1200
#define MULLO_BASECASE_THRESHOLD_LIMIT 200
#define SQRLO_BASECASE_THRESHOLD_LIMIT 200
#define SQRLO_DC_THRESHOLD_LIMIT 400
#define GET_STR_THRESHOLD_LIMIT 150
#define FAC_DSC_THRESHOLD_LIMIT 2048

#endif /* TUNE_PROGRAM_BUILD */

#if defined(__cplusplus)
    }
#endif

    /* FIXME: Make these itch functions less conservative.  Also consider making
       them dependent on just 'an', and compute the allocation directly from 'an'
       instead of via n.  */

    /* toom22/toom2: Scratch need is 2*(an + k), k is the recursion depth.
       k is ths smallest k such that
         ceil(an/2^k) < MUL_TOOM22_THRESHOLD.
       which implies that
         k = bitsize of floor ((an-1)/(MUL_TOOM22_THRESHOLD-1))
           = 1 + floor (log_2 (floor ((an-1)/(MUL_TOOM22_THRESHOLD-1))))
    */
#define gpmpn_toom22_mul_itch(an, bn) \
        (2 * ((an) + GMP_NUMB_BITS))
#define gpmpn_toom2_sqr_itch(an) \
        (2 * ((an) + GMP_NUMB_BITS))

    /* toom33/toom3: Scratch need is 5an/2 + 10k, k is the recursion depth.
       We use 3an + C, so that we can use a smaller constant.
     */
#define gpmpn_toom33_mul_itch(an, bn) \
        (3 * (an) + GMP_NUMB_BITS)
#define gpmpn_toom3_sqr_itch(an) \
        (3 * (an) + GMP_NUMB_BITS)

    /* toom33/toom3: Scratch need is 8an/3 + 13k, k is the recursion depth.
       We use 3an + C, so that we can use a smaller constant.
     */
#define gpmpn_toom44_mul_itch(an, bn) \
        (3 * (an) + GMP_NUMB_BITS)
#define gpmpn_toom4_sqr_itch(an) \
        (3 * (an) + GMP_NUMB_BITS)

#define gpmpn_toom6_sqr_itch(n)                         \
        (((n) - SQR_TOOM6_THRESHOLD) * 2 +                \
         MAX(SQR_TOOM6_THRESHOLD * 2 + GMP_NUMB_BITS * 6, \
             gpmpn_toom4_sqr_itch(SQR_TOOM6_THRESHOLD)))

#define MUL_TOOM6H_MIN \
        ((MUL_TOOM6H_THRESHOLD > MUL_TOOM44_THRESHOLD) ? MUL_TOOM6H_THRESHOLD : MUL_TOOM44_THRESHOLD)
#define gpmpn_toom6_mul_n_itch(n)                  \
        (((n) - MUL_TOOM6H_MIN) * 2 +                \
         MAX(MUL_TOOM6H_MIN * 2 + GMP_NUMB_BITS * 6, \
             gpmpn_toom44_mul_itch(MUL_TOOM6H_MIN, MUL_TOOM6H_MIN)))

    ANYCALLER static inline mp_size_t
    gpmpn_toom6h_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t estimatedN;
        estimatedN = (an + bn) / (size_t)10 + 1;
        return gpmpn_toom6_mul_n_itch(estimatedN * 6);
    }

#define gpmpn_toom8_sqr_itch(n)                                   \
        ((((n) * 15) >> 3) - ((SQR_TOOM8_THRESHOLD * 15) >> 3) +    \
         MAX(((SQR_TOOM8_THRESHOLD * 15) >> 3) + GMP_NUMB_BITS * 6, \
             gpmpn_toom6_sqr_itch(SQR_TOOM8_THRESHOLD)))

#define MUL_TOOM8H_MIN \
        ((MUL_TOOM8H_THRESHOLD > MUL_TOOM6H_MIN) ? MUL_TOOM8H_THRESHOLD : MUL_TOOM6H_MIN)
#define gpmpn_toom8_mul_n_itch(n)                            \
        ((((n) * 15) >> 3) - ((MUL_TOOM8H_MIN * 15) >> 3) +    \
         MAX(((MUL_TOOM8H_MIN * 15) >> 3) + GMP_NUMB_BITS * 6, \
             gpmpn_toom6_mul_n_itch(MUL_TOOM8H_MIN)))

    ANYCALLER static inline mp_size_t
    gpmpn_toom8h_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t estimatedN;
        estimatedN = (an + bn) / (size_t)14 + 1;
        return gpmpn_toom8_mul_n_itch(estimatedN * 8);
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom32_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = 1 + (2 * an >= 3 * bn ? (an - 1) / (size_t)3 : (bn - 1) >> 1);
        mp_size_t itch = 2 * n + 1;

        return itch;
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom42_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = an >= 2 * bn ? (an + 3) >> 2 : (bn + 1) >> 1;
        return 6 * n + 3;
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom43_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = 1 + (3 * an >= 4 * bn ? (an - 1) >> 2 : (bn - 1) / (size_t)3);

        return 6 * n + 4;
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom52_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = 1 + (2 * an >= 5 * bn ? (an - 1) / (size_t)5 : (bn - 1) >> 1);
        return 6 * n + 4;
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom53_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = 1 + (3 * an >= 5 * bn ? (an - 1) / (size_t)5 : (bn - 1) / (size_t)3);
        return 10 * n + 10;
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom62_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = 1 + (an >= 3 * bn ? (an - 1) / (size_t)6 : (bn - 1) >> 1);
        return 10 * n + 10;
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom63_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = 1 + (an >= 2 * bn ? (an - 1) / (size_t)6 : (bn - 1) / (size_t)3);
        return 9 * n + 3;
    }

    ANYCALLER static inline mp_size_t
    gpmpn_toom54_mul_itch(mp_size_t an, mp_size_t bn)
    {
        mp_size_t n = 1 + (4 * an >= 5 * bn ? (an - 1) / (size_t)5 : (bn - 1) / (size_t)4);
        return 9 * n + 3;
    }

    /* let S(n) = space required for input size n,
       then S(n) = 3 floor(n/2) + 1 + S(floor(n/2)).   */
#define gpmpn_toom42_mulmid_itch(n) \
        (3 * (n) + GMP_NUMB_BITS)

#if 0
#define gpmpn_fft_mul gpmpn_mul_fft_full
#else
#define gpmpn_fft_mul gpmpn_nussbaumer_mul
#endif

#ifdef __cplusplus

    /* A little helper for a null-terminated __gpgmp_allocate_func string.
       The destructor ensures it's freed even if an exception is thrown.
       The len field is needed by the destructor, and can be used by anyone else
       to avoid a second strlen pass over the data.

       Since our input is a C string, using strlen is correct.  Perhaps it'd be
       more C++-ish style to use std::char_traits<char>::length, but char_traits
       isn't available in gcc 2.95.4.  */

    class gmp_allocated_string
    {
    public:
        char *str;
        size_t len;
        gmp_allocated_string(char *arg)
        {
            str = arg;
            len = std::strlen(str);
        }
        ~gmp_allocated_string()
        {
            (*__gpgmp_free_func)(str, len + 1);
        }
    };

    std::istream &__gmpz_operator_in_nowhite(std::istream &, mpz_ptr, char);
    int __gmp_istream_set_base(std::istream &, char &, bool &, bool &);
    void __gmp_istream_set_digits(std::string &, std::istream &, char &, bool &, int);
    void __gmp_doprnt_params_from_ios(struct doprnt_params_t *, std::ios &);
    std::ostream &__gmp_doprnt_integer_ostream(std::ostream &, struct doprnt_params_t *, char *);
    extern const struct doprnt_funs_t __gmp_asprintf_funs_noformat;

#endif /* __cplusplus */

#endif /* __GPGMP_IMPL_H__ */
