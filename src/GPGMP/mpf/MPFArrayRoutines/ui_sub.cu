#include "GPGMP/gpgmp-impl.cuh"

namespace gpgmp
{
	namespace mpfArrayRoutines
	{

		ANYCALLER void
		mpf_ui_sub(mpf_array_idx r, unsigned long int u, mpf_array_idx v)
		{
			__mpf_struct uu;
			mp_limb_t ul;

			if (u == 0)
			{
				gpmpf_neg(r, v);
				return;
			}

			ul = u;
			uu._mp_size = 1;
			uu._mp_d = &ul;
			uu._mp_exp = 1;

			//doing -(v - uu) is functionally equivalent to uu - v;
			//allows us to reuse the sub_mpf_t_from_array_idx routine.
			gpgmp::internal::mpfArrayRoutines::gpmpf_sub_mpf_t_from_array_idx(r, v, &uu);
			gpmpf_neg(r, r);
		}

	}
}