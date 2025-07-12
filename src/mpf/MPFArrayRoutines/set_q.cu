#include "gpgmp-impl.cuh"

namespace gpgmp
{
  namespace mpfArrayRoutines
  {

    /* As usual the aim is to produce PREC(r) limbs, with the high non-zero.  The
       basic mpn_div_q produces a quotient of nsize-dsize+1 limbs, with either the
       high or second highest limb non-zero.  We arrange for nsize-dsize+1 to equal
       prec+1, hence giving either prec or prec+1 result limbs at PTR(r).

       nsize-dsize+1 == prec+1 is achieved by adjusting num(q), either dropping low
       limbs if it's too big, or padding with low zeros if it's too small.  The
       full given den(q) is always used.

       We cannot truncate den(q), because even when it's much bigger than prec the
       last limbs can still influence the final quotient.  Often they don't, but we
       leave optimization of that to mpn_div_q.

       Enhancements:

       The high quotient limb is non-zero when high{np,dsize} > {dp,dsize}.  We
       could make that comparison and use qsize==prec instead of qsize==prec+1,
       to save one limb in the division.  */

    //TODO: Refactor to avoid dynamic allocation.
    ANYCALLER void
    gpmpf_set_q(mpf_array_idx r, mpq_srcptr q)
    {
      mp_srcptr np, dp;
      mp_size_t prec, nsize, dsize, qsize, prospective_qsize, tsize, zeros;
      mp_size_t sign_quotient, high_zero;
      mp_ptr qp, tp;
      mp_exp_t exp;
      TMP_DECL;

      ASSERT(SIZ(&q->_mp_den) > 0); /* canonical q */

      nsize = SIZ(&q->_mp_num);
      dsize = SIZ(&q->_mp_den);

      if (UNLIKELY(nsize == 0))
      {
        MPF_ARRAY_SIZES(r.array)[r.idx] = 0;
        MPF_ARRAY_EXPONENTS(r.array)[r.idx] = 0;
        return;
      }

      TMP_MARK;

      prec = r.array->userSpecifiedPrecisionLimbCount;
      qp = MPF_ARRAY_DATA_AT_IDX(r.array, r.idx);

      sign_quotient = nsize;
      nsize = ABS(nsize);
      np = PTR(&q->_mp_num);
      dp = PTR(&q->_mp_den);

      prospective_qsize = nsize - dsize + 1; /* q from using given n,d sizes */
      exp = prospective_qsize;               /* ie. number of integer limbs */
      qsize = prec + 1;                      /* desired q */

      zeros = qsize - prospective_qsize; /* n zeros to get desired qsize */
      tsize = nsize + zeros;             /* size of intermediate numerator */
      tp = TMP_ALLOC_LIMBS(tsize + 1);   /* +1 for mpn_div_q's scratch */

      if (zeros > 0)
      {
        /* pad n with zeros into temporary space */
        MPN_ZERO(tp, zeros);
        MPN_COPY(tp + zeros, np, nsize);
        np = tp; /* mpn_div_q allows this overlap */
      }
      else
      {
        /* shorten n to get desired qsize */
        np -= zeros;
      }

      ASSERT(tsize - dsize + 1 == qsize);
      gpgmp::mpnRoutines::gpmpn_div_q(qp, np, tsize, dp, dsize, tp);

      /* strip possible zero high limb */
      high_zero = (qp[qsize - 1] == 0);
      qsize -= high_zero;
      exp -= high_zero;

      MPF_ARRAY_EXPONENTS(r.array)[r.idx] = exp;
      MPF_ARRAY_SIZES(r.array)[r.idx] = sign_quotient >= 0 ? qsize : -qsize;

      TMP_FREE;
    }

  }
}