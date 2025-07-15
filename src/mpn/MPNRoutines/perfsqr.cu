/* gpmpn_perfect_square_p(u,usize) -- Return non-zero if U is a perfect square,
   zero otherwise.

Copyright 1991, 1993, 1994, 1996, 1997, 2000-2002, 2005, 2012 Free Software
Foundation, Inc.

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

#include <stdio.h> /* for NULL */
#include "gpgmp-impl.cuh"
#include "longlong.cuh"

// PERFSQR.H
#if GMP_LIMB_BITS != 64 || GMP_NAIL_BITS != 0
Error, error, this data is for 64 bit limb and 0 bit nail
#endif

#ifdef __CUDA_ARCH__
/* Non-zero bit indicates a quadratic residue mod 0x100.
   This test identifies 82.81% as non-squares (212/256). */
__device__ static const mp_limb_t
sq_res_0x100[4] = {
  CNST_LIMB(0x202021202030213),
  CNST_LIMB(0x202021202020213),
  CNST_LIMB(0x202021202030212),
  CNST_LIMB(0x202021202020212),
};
#else
/* Non-zero bit indicates a quadratic residue mod 0x100.
   This test identifies 82.81% as non-squares (212/256). */
static const mp_limb_t
sq_res_0x100[4] = {
  CNST_LIMB(0x202021202030213),
  CNST_LIMB(0x202021202020213),
  CNST_LIMB(0x202021202030212),
  CNST_LIMB(0x202021202020212),
};
#endif

/* 2^48-1 = 3^2 * 5 * 7 * 13 * 17 * 97 ... */
#define PERFSQR_MOD_BITS  49

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


namespace gpgmp
{
  namespace mpnRoutines
  {

/* change this to "#define TRACE(x) x" for diagnostics */
#define TRACE(x)

    /* PERFSQR_MOD_* detects non-squares using residue tests.

       A macro PERFSQR_MOD_TEST is setup by gen-psqr.c in perfsqr.h.  It takes
       {up,usize} modulo a selected modulus to get a remainder r.  For 32-bit or
       64-bit limbs this modulus will be 2^24-1 or 2^48-1 using PERFSQR_MOD_34,
       or for other limb or nail sizes a PERFSQR_PP is chosen and PERFSQR_MOD_PP
       used.  PERFSQR_PP_NORM and PERFSQR_PP_INVERTED are pre-calculated in this
       case too.

       PERFSQR_MOD_TEST then makes various calls to PERFSQR_MOD_1 or
       PERFSQR_MOD_2 with divisors d which are factors of the modulus, and table
       data indicating residues and non-residues modulo those divisors.  The
       table data is in 1 or 2 limbs worth of bits respectively, per the size of
       each d.

       A "modexact" style remainder is taken to reduce r modulo d.
       PERFSQR_MOD_IDX implements this, producing an index "idx" for use with
       the table data.  Notice there's just one multiplication by a constant
       "inv", for each d.

       The modexact doesn't produce a true r%d remainder, instead idx satisfies
       "-(idx<<PERFSQR_MOD_BITS) == r mod d".  Because d is odd, this factor
       -2^PERFSQR_MOD_BITS is a one-to-one mapping between r and idx, and is
       accounted for by having the table data suitably permuted.

       The remainder r fits within PERFSQR_MOD_BITS which is less than a limb.
       In fact the GMP_LIMB_BITS - PERFSQR_MOD_BITS spare bits are enough to fit
       each divisor d meaning the modexact multiply can take place entirely
       within one limb, giving the compiler the chance to optimize it, in a way
       that say umul_ppmm would not give.

       There's no need for the divisors d to be prime, in fact gen-psqr.c makes
       a deliberate effort to combine factors so as to reduce the number of
       separate tests done on r.  But such combining is limited to d <=
       2*GMP_LIMB_BITS so that the table data fits in at most 2 limbs.

       Alternatives:

       It'd be possible to use bigger divisors d, and more than 2 limbs of table
       data, but this doesn't look like it would be of much help to the prime
       factors in the usual moduli 2^24-1 or 2^48-1.

       The moduli 2^24-1 or 2^48-1 are nothing particularly special, they're
       just easy to calculate (see gpmpn_mod_34lsub1) and have a nice set of prime
       factors.  2^32-1 and 2^64-1 would be equally easy to calculate, but have
       fewer prime factors.

       The nails case usually ends up using gpmpn_mod_1, which is a lot slower
       than gpmpn_mod_34lsub1.  Perhaps other such special moduli could be found
       for the nails case.  Two-term things like 2^30-2^15-1 might be
       candidates.  Or at worst some on-the-fly de-nailing would allow the plain
       2^24-1 to be used.  Currently nails are too preliminary to be worried
       about.

    */

#define PERFSQR_MOD_MASK ((CNST_LIMB(1) << PERFSQR_MOD_BITS) - 1)

#define MOD34_BITS (GMP_NUMB_BITS / 4 * 3)
#define MOD34_MASK ((CNST_LIMB(1) << MOD34_BITS) - 1)

#define PERFSQR_MOD_34(r, up, usize)                \
  do                                                \
  {                                                 \
    (r) = gpmpn_mod_34lsub1(up, usize);               \
    (r) = ((r) & MOD34_MASK) + ((r) >> MOD34_BITS); \
  } while (0)

/* FIXME: The %= here isn't good, and might destroy any savings from keeping
   the PERFSQR_MOD_IDX stuff within a limb (rather than needing umul_ppmm).
   Maybe a new sort of gpmpn_preinv_mod_1 could accept an unnormalized divisor
   and a shift count, like gpmpn_preinv_divrem_1.  But mod_34lsub1 is our
   normal case, so lets not worry too much about mod_1.  */
#define PERFSQR_MOD_PP(r, up, usize)                             \
  do                                                             \
  {                                                              \
    if (BELOW_THRESHOLD(usize, PREINV_MOD_1_TO_MOD_1_THRESHOLD)) \
    {                                                            \
      (r) = gpmpn_preinv_mod_1(up, usize, PERFSQR_PP_NORM,         \
                             PERFSQR_PP_INVERTED);               \
      (r) %= PERFSQR_PP;                                         \
    }                                                            \
    else                                                         \
    {                                                            \
      (r) = gpmpn_mod_1(up, usize, PERFSQR_PP);                    \
    }                                                            \
  } while (0)

#define PERFSQR_MOD_IDX(idx, r, d, inv)              \
  do                                                 \
  {                                                  \
    mp_limb_t q;                                     \
    ASSERT((r) <= PERFSQR_MOD_MASK);                 \
    ASSERT((((inv) * (d)) & PERFSQR_MOD_MASK) == 1); \
    ASSERT(MP_LIMB_T_MAX / (d) >= PERFSQR_MOD_MASK); \
                                                     \
    q = ((r) * (inv)) & PERFSQR_MOD_MASK;            \
    ASSERT(r == ((q * (d)) & PERFSQR_MOD_MASK));     \
    (idx) = (q * (d)) >> PERFSQR_MOD_BITS;           \
  } while (0)

#define PERFSQR_MOD_1(r, d, inv, mask)                  \
  do                                                    \
  {                                                     \
    unsigned idx;                                       \
    ASSERT((d) <= GMP_LIMB_BITS);                       \
    PERFSQR_MOD_IDX(idx, r, d, inv);                    \
    TRACE(printf("  PERFSQR_MOD_1 d=%u r=%lu idx=%u\n", \
                 d, r % d, idx));                       \
    if ((((mask) >> idx) & 1) == 0)                     \
    {                                                   \
      TRACE(printf("  non-square\n"));                  \
      return 0;                                         \
    }                                                   \
  } while (0)

/* The expression "(int) idx - GMP_LIMB_BITS < 0" lets the compiler use the
   sign bit from "idx-GMP_LIMB_BITS", which might help avoid a branch.  */
#define PERFSQR_MOD_2(r, d, inv, mhi, mlo)              \
  do                                                    \
  {                                                     \
    mp_limb_t m;                                        \
    unsigned idx;                                       \
    ASSERT((d) <= 2 * GMP_LIMB_BITS);                   \
                                                        \
    PERFSQR_MOD_IDX(idx, r, d, inv);                    \
    TRACE(printf("  PERFSQR_MOD_2 d=%u r=%lu idx=%u\n", \
                 d, r % d, idx));                       \
    m = ((int)idx - GMP_LIMB_BITS < 0 ? (mlo) : (mhi)); \
    idx %= GMP_LIMB_BITS;                               \
    if (((m >> idx) & 1) == 0)                          \
    {                                                   \
      TRACE(printf("  non-square\n"));                  \
      return 0;                                         \
    }                                                   \
  } while (0)

    HOSTONLY int gpmpn_perfect_square_p(mp_srcptr up, mp_size_t usize)
    {
      ASSERT(usize >= 1);

      TRACE(gmp_printf("gpmpn_perfect_square_p %Nd\n", up, usize));

      /* The first test excludes 212/256 (82.8%) of the perfect square candidates
         in O(1) time.  */
      {
        unsigned idx = up[0] % 0x100;
        if (((sq_res_0x100[idx / GMP_LIMB_BITS] >> (idx % GMP_LIMB_BITS)) & 1) == 0)
          return 0;
      }

#if 0
  /* Check that we have even multiplicity of 2, and then check that the rest is
     a possible perfect square.  Leave disabled until we can determine this
     really is an improvement.  If it is, it could completely replace the
     simple probe above, since this should throw out more non-squares, but at
     the expense of somewhat more cycles.  */
  {
    mp_limb_t lo;
    int cnt;
    lo = up[0];
    while (lo == 0)
      up++, lo = up[0], usize--;
    count_trailing_zeros (cnt, lo);
    if ((cnt & 1) != 0)
      return 0;			/* return of not even multiplicity of 2 */
    lo >>= cnt;			/* shift down to align lowest non-zero bit */
    if ((lo & 6) != 0)
      return 0;
  }
#endif

      /* The second test uses gpmpn_mod_34lsub1 or gpmpn_mod_1 to detect non-squares
         according to their residues modulo small primes (or powers of
         primes).  See perfsqr.h.  */
      PERFSQR_MOD_TEST(up, usize);

      /* For the third and last test, we finally compute the square root,
         to make sure we've really got a perfect square.  */
      {
        mp_ptr root_ptr;
        int res;
        TMP_DECL;

        TMP_MARK;
        root_ptr = TMP_ALLOC_LIMBS((usize + 1) / 2);

        /* Iff gpmpn_sqrtrem returns zero, the square is perfect.  */
        res = !gpmpn_sqrtrem(root_ptr, NULL, up, usize);
        TMP_FREE;

        return res;
      }
    }

  }
}