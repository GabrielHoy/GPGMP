#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* The precision we use for d = x-y is based on what mpf_div will want from
       the dividend.  It calls mpn_div_q to produce a quotient of rprec+1 limbs.
       So rprec+1 == dsize - xsize + 1, hence dprec = rprec+xsize.  */

    ANYCALLER void
    gpmpf_reldiff(mpf_array_idx rdiff, mpf_array_idx x, mpf_array_idx y)
    {
      if (UNLIKELY(MPF_ARRAY_SIZES(x.array)[x.idx] == 0))
      {
        gpmpf_set_ui(rdiff, (unsigned long int)(mpf_array_sgn(y) != 0));
      }
      else
      {
        mp_limb_t* scratchSpace = MPF_ARRAY_SCRATCH_SPACE_FOR_IDX(rdiff.array, rdiff.idx);
        mp_size_t dprec;
        mpf_t d;

        dprec = rdiff.array->userSpecifiedPrecisionLimbCount + ABS(MPF_ARRAY_SIZES(x.array)[x.idx]);
        ASSERT(rdiff.array->userSpecifiedPrecisionLimbCount + 1 == dprec - ABS(MPF_ARRAY_SIZES(x.array)[x.idx]) + 1);

        PREC(d) = dprec;
        PTR(d) = scratchSpace;
        scratchSpace += dprec;

        gpgmp::internal::mpfRoutines::gpmpf_sub_mpf_array_idx_from_mpf_array_idx(d, x, y, scratchSpace); //gpmpf_sub(d, x, y, scratchSpace);
        SIZ(d) = ABSIZ(d);
        gpgmp::internal::mpfArrayRoutines::gpmpf_div_mpf_t_by_mpf_array_idx(rdiff, d, x, scratchSpace); //gpmpf_div(rdiff, d, x, scratchSpace);

      }
    }

  }
}