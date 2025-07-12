#include "gpgmp-impl.cuh"
#include "longlong.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* The core operation is a multiply of PREC(r) limbs from u by v, producing
       either PREC(r) or PREC(r)+1 result limbs.  If u is shorter than PREC(r),
       then we take only as much as it has.  If u is longer we incorporate a
       carry from the lower limbs.

       If u has just 1 extra limb, then the carry to add is high(up[0]*v).  That
       is of course what mpn_mul_1 would do if it was called with PREC(r)+1
       limbs of input.

       If u has more than 1 extra limb, then there can be a further carry bit
       out of lower uncalculated limbs (the way the low of one product adds to
       the high of the product below it).  This is of course what an mpn_mul_1
       would do if it was called with the full u operand.  But we instead work
       downwards explicitly, until a carry occurs or until a value other than
       GMP_NUMB_MAX occurs (that being the only value a carry bit can propagate
       across).

       The carry determination normally requires two umul_ppmm's, only rarely
       will GMP_NUMB_MAX occur and require further products.

       The carry limb is conveniently added into the mul_1 using mpn_mul_1c when
       that function exists, otherwise a subsequent mpn_add_1 is needed.

       Clearly when mpn_mul_1c is used the carry must be calculated first.  But
       this is also the case when add_1 is used, since if r==u and ABSIZ(r) >
       PREC(r) then the mpn_mul_1 overwrites the low part of the input.

       A reuse r==u with size > prec can occur from a size PREC(r)+1 in the
       usual way, or it can occur from an mpf_set_prec_raw leaving a bigger
       sized value.  In both cases we can end up calling mpn_mul_1 with
       overlapping src and dst regions, but this will be with dst < src and such
       an overlap is permitted.

       Not done:

       No attempt is made to determine in advance whether the result will be
       PREC(r) or PREC(r)+1 limbs.  If it's going to be PREC(r)+1 then we could
       take one less limb from u and generate just PREC(r), that of course
       satisfying application requested precision.  But any test counting bits
       or forming the high product would almost certainly take longer than the
       incremental cost of an extra limb in mpn_mul_1.

       Enhancements:

       Repeated mpf_mul_ui's with an even v will accumulate low zero bits on the
       result, leaving low zero limbs after a while, which it might be nice to
       strip to save work in subsequent operations.  Calculating the low limb
       explicitly would let us direct mpn_mul_1 to put the balance at rp when
       the low is zero (instead of normally rp+1).  But it's not clear whether
       this would be worthwhile.  Explicit code for the low limb will probably
       be slower than having it done in mpn_mul_1, so we need to consider how
       often a zero will be stripped and how much that's likely to save
       later.  */

    ANYCALLER void
    gpmpf_mul_ui(mpf_array_idx r, mpf_array_idx u, unsigned long int v)
    {
      mp_srcptr up;
      mp_size_t usize;
      mp_size_t size;
      mp_size_t prec, excess;
      mp_limb_t cy_limb, vl, cbit, cin;
      mp_ptr rp;

      usize = MPF_ARRAY_SIZES(u.array)[u.idx];
      if (UNLIKELY(v == 0) || UNLIKELY(usize == 0))
      {
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      //TODO: Handle this case
#if BITS_PER_ULONG > GMP_NUMB_BITS /* avoid warnings about shift amount */
      #error "BITS_PER_ULONG > GMP_NUMB_BITS, this case is not supported in GPGMP yet..."
      if (v > GMP_NUMB_MAX)
      {
        mpf_t vf;
        mp_limb_t vp[2];
        vp[0] = v & GMP_NUMB_MASK;
        vp[1] = v >> GMP_NUMB_BITS;
        PTR(vf) = vp;
        SIZ(vf) = 2;
        ASSERT_CODE(PREC(vf) = 2);
        EXP(vf) = 2;
        gpmpf_mul(r, u, vf);
        return;
      }
#endif

      size = ABS(usize);
      prec = r.array->userSpecifiedPrecisionLimbCount;
      up = MPF_ARRAY_DATA_AT_IDX(u.array, u.idx);
      vl = v;
      excess = size - prec;
      cin = 0;

      if (excess > 0)
      {
        /* up is bigger than desired rp, shorten it to prec limbs and
           determine a carry-in */

        mp_limb_t vl_shifted = vl << GMP_NAIL_BITS;
        mp_limb_t hi, lo, next_lo, sum;
        mp_size_t i;

        /* high limb of top product */
        i = excess - 1;
        umul_ppmm(cin, lo, up[i], vl_shifted);

        /* and carry bit out of products below that, if any */
        for (;;)
        {
          i--;
          if (i < 0)
            break;

          umul_ppmm(hi, next_lo, up[i], vl_shifted);
          lo >>= GMP_NAIL_BITS;
          ADDC_LIMB(cbit, sum, hi, lo);
          cin += cbit;
          lo = next_lo;

          /* Continue only if the sum is GMP_NUMB_MAX.  GMP_NUMB_MAX is the
             only value a carry from below can propagate across.  If we've
             just seen the carry out (ie. cbit!=0) then sum!=GMP_NUMB_MAX,
             so this test stops us for that case too.  */
          if (LIKELY(sum != GMP_NUMB_MAX))
            break;
        }

        up += excess;
        size = prec;
      }

      rp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);
#if HAVE_NATIVE_mpn_mul_1c
      cy_limb = mpn_mul_1c(rp, up, size, vl, cin);
#else
      cy_limb = gpgmp::mpnRoutines::gpmpn_mul_1(rp, up, size, vl);
      __GMPN_ADD_1(cbit, rp, rp, size, cin);
      cy_limb += cbit;
#endif
      rp[size] = cy_limb;
      cy_limb = cy_limb != 0;
      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = MPF_ARRAY_EXPONENTS(u.array)[u.idx] + cy_limb;
      size += cy_limb;
      MPF_ARRAY_SIZES(r.array)[r.idx] = usize >= 0 ? size : -size;
    }

  }
}