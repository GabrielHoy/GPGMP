//"Wrapper" header meant to mimic gmp.h
#pragma once
#include "gmp.cuh"
#include "Definitions.cuh"

// GPGMP's equivalent of __GPGMP_DECLSPEC.
#define __GPGMP_DECLSPEC
#define __GPGMP_CALLERTYPE ANYCALLER
#define __GPGMP_MPN(x) __gpgmpn_##x


namespace gpgmp
{

        enum UsedOperationFlags
            {
                // Ignore...
                    OP_NONE = 0,
                    // Addition between two floating-point numbers.
                    OP_ADD = 1 << 0,
                    // Subtraction between two floating-point numbers.
                    OP_SUB = 1 << 1,
                    // Multiplication between two floating-point numbers.
                    OP_MUL = 1 << 2,
                    // Division between two floating-point numbers.
                    OP_DIV = 1 << 3,
                    // Division of unsigned integers by floating-point numbers.
                    OP_UI_DIV = 1 << 4,
                    // Division of floating-point numbers by unsigned integers.
                    OP_DIV_UI = 1 << 5,
                    // Square root of a floating-point number.
                    OP_SQRT = 1 << 6,
                    // Square root of an unsigned integer.
                    OP_SQRT_UI = 1 << 7,
                    // Absolute difference between two floating-point numbers.
                    OP_RELDIFF = 1 << 8,
                    // Ensure all operations listed in this enum are available; this will allocate a lot of extra memory for scratch space, caution is advised...
                    OP_ALL = UINT32_MAX
            };

            // A direct alias for uint32_t, used to identify the enum type used inside of available operation bit-fields in a more readable manner.
            typedef uint32_t _UsedOperationFlags;

            struct __align__(128) mpf_array
            {
                    int numFloatsInArray;                    // Number of floating-point numbers that this array contains.
                    int userSpecifiedPrecisionLimbCount;     // Number of limbs that the user specified for the precision of each floating-point number in this array.
                    int limbsPerArrayFloat;                  // Number of limbs *actually allocated* for each floating-point number in this array to use - this accounts for any "scratch space" limbs.
                    _UsedOperationFlags availableOperations; // Bitfield of UsedOperationFlags - Flags specifying which operations are available for use on a given floating-point number.

                    //...Also contains arrays immediately following the struct similar to mpn_array - these arrays are:
                    //- "_mp_sizes" - an array of SIGNED ints, representing how many limbs are used by each float in the array.
                    //- "_mp_exponents" - an array of mp_exp_t's(usually typedefed long int's) - representing the exponent of each float in the array.
                    //- "_mp_data" - an array - of arrays - of mp_limb_t's, flattened out into 1D - representing the total limb data for each float in the array. i.e an mpf_array[2] with a precision of 2 limbs could have a data array that reads {0,42,1,0}, thereby arr[0] = ((2^64)*0) + 42 and arr[1] = ((2^64)*1) + 0
                    //
                    // Important notes about the "_mp_data" array that differs from GMP behavior:
                    // Usually GMP "lies" about the number of limbs that are allocated by a float and adds one to the actual desired precision from the user; they however still store the desired, lower precision value inside of _mp_prec.
                    // We replicate most of this behavior, but we more directly expose the actual precision that the float "has" by storing the "value+1" count of limbs inside limbsPerArrayFloat, instead of storing the user-specified number and just allocating n+1 limbs.
                    // This means that when implementing any formulas you can simply use limbsPerArrayFloat as the source-of-truth for stride values etc, without needing to worry about this "trailing" limb interfering with any of your pointer offsets, etc.

                    // Helper operators for array indexing so that users can use common syntax like arr[0], arr[1], arr[2]...
                    ANYCALLER mp_limb_t &operator[](int index)
                    {
                            return *(MPF_ARRAY_DATA(this) + (index * this->limbsPerArrayFloat));
                    }

                    ANYCALLER const mp_limb_t &operator[](int index) const
                    {
                            return *(MPF_ARRAY_DATA_CONST(this) + (index * this->limbsPerArrayFloat));
                    }
            };

            //mpf_array_idx's are simple structs pairing an mpf_array* and an int, representing an index in a array.
            //Nearly all routines having to do with mpf_array's will use this struct to represent their input parameters, since they need a reference to both the array and the index in it.
            struct mpf_array_idx
            {
                    mpf_array *array;
                    int idx;
            };


        //This is a direct alias for an mpf_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpf_array is on the host or the device.
        typedef mpf_array* mpf_device_array;
        //This is a direct alias for an mpf_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpf_array is on the host or the device.
        typedef mpf_array* mpf_host_array;


        /**************** Float (i.e. F) routines.  ****************/
        namespace mpfRoutines
        {
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_abs(mpf_ptr, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_add(mpf_ptr, mpf_srcptr, mpf_srcptr, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_add_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_add_ui(mpf_ptr, mpf_srcptr, unsigned long int, mp_limb_t*);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_ceil(mpf_ptr, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_clear(mpf_ptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_clears(mpf_ptr, ...);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp(mpf_srcptr, mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_z(mpf_srcptr, mpz_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_d(mpf_srcptr, double) __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_si(mpf_srcptr, signed long int) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_ui(mpf_srcptr, unsigned long int) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_div(mpf_ptr, mpf_srcptr, mpf_srcptr, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_div_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_div_2exp(mpf_ptr, mpf_srcptr, mp_bitcnt_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_div_ui(mpf_ptr, mpf_srcptr, unsigned long int, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_div_ui_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_dump(mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_eq(mpf_srcptr, mpf_srcptr, mp_bitcnt_t) __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_sint_p(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_slong_p(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_sshort_p(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_uint_p(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_ulong_p(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_ushort_p(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_floor(mpf_ptr, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE double gpmpf_get_d(mpf_srcptr) __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE double gpmpf_get_d_2exp(signed long int *, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpf_get_default_prec(void) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpf_get_prec(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE long gpmpf_get_si(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE char *gpmpf_get_str(char *, mp_exp_t *, int, size_t, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE unsigned long gpmpf_get_ui(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_init(mpf_ptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_init2(mpf_ptr, mp_bitcnt_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_inits(mpf_ptr, ...);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_init_set(mpf_ptr, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_init_set_d(mpf_ptr, double);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_init_set_si(mpf_ptr, signed long int);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_init_set_str(mpf_ptr, const char *, int);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_init_set_ui(mpf_ptr, unsigned long int);

    #ifdef _GMP_H_HAVE_FILE
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpf_inp_str(mpf_ptr, FILE *, int);
    #endif

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_integer_p(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul(mpf_ptr, mpf_srcptr, mpf_srcptr, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_mul_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul_2exp(mpf_ptr, mpf_srcptr, mp_bitcnt_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul_ui(mpf_ptr, mpf_srcptr, unsigned long int);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_neg(mpf_ptr, mpf_srcptr);

    #ifdef _GMP_H_HAVE_FILE
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpf_out_str(FILE *, int, size_t, mpf_srcptr);
    #endif

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_pow_ui(mpf_ptr, mpf_srcptr, unsigned long int, mp_limb_t*);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_random2(mpf_ptr, mp_size_t, mp_exp_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_reldiff(mpf_ptr, mpf_srcptr, mpf_srcptr, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_reldiff_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set(mpf_ptr, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_d(mpf_ptr, double);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_default_prec(mp_bitcnt_t) __GMP_NOTHROW;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_prec(mpf_ptr, mp_bitcnt_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_prec_raw(mpf_ptr, mp_bitcnt_t) __GMP_NOTHROW;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_q(mpf_ptr, mpq_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_si(mpf_ptr, signed long int);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_set_str(mpf_ptr, const char *, int);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_ui(mpf_ptr, unsigned long int);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_z(mpf_ptr, mpz_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpf_size(mpf_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sqrt(mpf_ptr, mpf_srcptr, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_sqrt_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sqrt_ui(mpf_ptr, unsigned long int, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_sqrt_ui_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sub(mpf_ptr, mpf_srcptr, mpf_srcptr, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_sub_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sub_ui(mpf_ptr, mpf_srcptr, unsigned long int, mp_limb_t*);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_swap(mpf_ptr, mpf_ptr) __GMP_NOTHROW;

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_trunc(mpf_ptr, mpf_srcptr);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_ui_div(mpf_ptr, unsigned long int, mpf_srcptr, mp_limb_t*);
            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE static int gpmpf_ui_div_itch(mp_size_t);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_ui_sub(mpf_ptr, unsigned long int, mpf_srcptr, mp_limb_t*);

            __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_urandomb(mpf_ptr, gmp_randstate_ptr, mp_bitcnt_t);

        }


        /**************** Float Array (i.e. mpf_array) routines.  ****************/
    namespace mpfArrayRoutines
    {

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_abs(mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_add(mpf_array_idx, mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_add_ui(mpf_array_idx, mpf_array_idx, unsigned long int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_ceil(mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp(mpf_array_idx, mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_z(mpf_array_idx, mpz_srcptr) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_d(mpf_array_idx, double) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_si(mpf_array_idx, signed long int) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_ui(mpf_array_idx, unsigned long int) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_div(mpf_array_idx, mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_div_2exp(mpf_array_idx, mpf_array_idx, mp_bitcnt_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_div_ui(mpf_array_idx, mpf_array_idx, unsigned long int);

        __GPGMP_DECLSPEC HOSTONLY void gpmpf_dump(mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_eq(mpf_array_idx, mpf_array_idx, mp_bitcnt_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_sint_p(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_slong_p(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_sshort_p(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_uint_p(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_ulong_p(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_fits_ushort_p(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_floor(mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE double gpmpf_get_d(mpf_array_idx) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE double gpmpf_get_d_2exp(signed long int *, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpf_get_prec(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE long gpmpf_get_si(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC HOSTONLY char *gpmpf_get_str(char *, mp_exp_t *, int, size_t, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE unsigned long gpmpf_get_ui(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

#ifdef _GMP_H_HAVE_FILE
        __GPGMP_DECLSPEC HOSTONLY size_t gpmpf_inp_str(mpf_array_idx, FILE *, int);
#endif

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_integer_p(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul(mpf_array_idx, mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul_2exp(mpf_array_idx, mpf_array_idx, mp_bitcnt_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul_ui(mpf_array_idx, mpf_array_idx, unsigned long int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_neg(mpf_array_idx, mpf_array_idx);

#ifdef _GMP_H_HAVE_FILE
        __GPGMP_DECLSPEC HOSTONLY size_t gpmpf_out_str(FILE *, int, size_t, mpf_array_idx);
#endif

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_pow_ui(mpf_array_idx, mpf_array_idx, unsigned long int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_random2(mpf_array_idx, mp_size_t, mp_exp_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_reldiff(mpf_array_idx, mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set(mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_d(mpf_array_idx, double);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_q(mpf_array_idx, mpq_srcptr);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_si(mpf_array_idx, signed long int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_set_str(mpf_array_idx, const char *, int, char*);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_ui(mpf_array_idx, unsigned long int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_z(mpf_array_idx, mpz_srcptr);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpf_size(mpf_array_idx) __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sqrt(mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sqrt_ui(mpf_array_idx, unsigned long int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sub(mpf_array_idx, mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sub_ui(mpf_array_idx, mpf_array_idx, unsigned long int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_swap(mpf_array_idx, mpf_array_idx) __GMP_NOTHROW;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_trunc(mpf_array_idx, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_ui_div(mpf_array_idx, unsigned long int, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_ui_sub(mpf_array_idx, unsigned long int, mpf_array_idx);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_urandomb(mpf_array_idx, gmp_randstate_ptr, mp_bitcnt_t);
    }

    namespace internal {

        //These primarily exist to allow for user-facing mpfArrayRoutines to be used in conjunction with mpf_t objects.
        namespace mpfArrayRoutines
        {
                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sub_mpf_t_from_array_idx(mpf_array_idx, mpf_array_idx, mpf_srcptr);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_add_mpf_t_to_array_idx(mpf_array_idx, mpf_array_idx, mpf_srcptr);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul_mpf_t_by_mpf_t(mpf_array_idx, mpf_srcptr, mpf_srcptr, mp_limb_t*);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul_mpf_t_by_mpf_array_idx(mpf_array_idx, mpf_srcptr, mpf_array_idx, mp_limb_t*);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_div_mpf_t_by_mpf_array_idx(mpf_array_idx, mpf_srcptr, mpf_array_idx, mp_limb_t*);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_array_idx_to_mpf_t(mpf_array_idx, mpf_srcptr);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_neg_set_array_idx_from_mpf_t(mpf_array_idx, mpf_srcptr);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpf_cmp_array_idx_to_mpf_t(mpf_array_idx, mpf_srcptr) __GMP_NOTHROW;
        }

        //These primarily exist to allow for user-facing mpfRoutines to be used in conjunction with mpf_t objects.
        namespace mpfRoutines
        {
                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_add_mpf_array_idx_to_mpf_array_idx(mpf_ptr, mpf_array_idx, mpf_array_idx, mp_limb_t*);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_sub_mpf_array_idx_from_mpf_array_idx(mpf_ptr, mpf_array_idx, mpf_array_idx, mp_limb_t*);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_set_mpf_t_to_array_idx(mpf_ptr, mpf_array_idx);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_mul_mpf_t_by_mpf_array_idx(mpf_ptr, mpf_srcptr, mpf_array_idx, mp_limb_t*);

                __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpf_neg_mpf_array_idx(mpf_ptr, mpf_array_idx);
        }

    }


    /************ Low level positive-integer (i.e. N) routines.  ************/
    namespace mpnRoutines
    {
/* This is ugly, but we need to make user calls reach the prefixed function. */
#define __GMP_FORCE_gpmpn_add 1
#define __GMP_FORCE_gpmpn_cmp 1
#define __GMP_FORCE_gpmpn_neg 1
#define __GMP_FORCE_gpmpn_sub_1 1
#define __GMP_FORCE_gpmpn_sub 1
#define __GMP_FORCE_gpmpn_zero_p 1
#define __GMP_FORCE_gpmpn_add_1 1

#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_add)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#endif

#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_add_1)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t) __GMP_NOTHROW;
#endif

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_add_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_addmul_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_cmp)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_cmp(mp_srcptr, mp_srcptr, mp_size_t)
//__GMP_NOTHROW __GMP_ATTRIBUTE_PURE;
#endif

#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_zero_p)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_zero_p(mp_srcptr, mp_size_t)
//__GMP_NOTHROW __GMP_ATTRIBUTE_PURE;
#endif

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_divexact_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_divexact_by3(dst, src, size) \
    gpmpn_divexact_by3c(dst, src, size, __GMP_CAST(mp_limb_t, 0))

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divexact_by3c(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define gpmpn_divmod_1(qp, np, nsize, dlimb) \
    gpmpn_divrem_1(qp, __GMP_CAST(mp_size_t, 0), np, nsize, dlimb)

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divrem(mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divrem_1(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_divrem_2(mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_div_qr_1(mp_ptr, mp_limb_t *, mp_srcptr, mp_size_t, mp_limb_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_div_qr_2(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_gcd(mp_ptr, mp_ptr, mp_size_t, mp_ptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_gcd_11(mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_gcd_1(mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_gcdext_1(mp_limb_signed_t *, mp_limb_signed_t *, mp_limb_t, mp_limb_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_gcdext(mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_size_t, mp_ptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpn_get_str(unsigned char *, int, mp_ptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_hamdist(mp_srcptr, mp_srcptr, mp_size_t)
            __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_lshift(mp_ptr, mp_srcptr, mp_size_t, unsigned int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mod_1(mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_mul_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_mul_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sqr(mp_ptr, mp_srcptr, mp_size_t);

#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_neg)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_neg(mp_ptr, mp_srcptr, mp_size_t);
#endif

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_com(mp_ptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_perfect_square_p(mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_perfect_power_p(mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_popcount(mp_srcptr, mp_size_t)
            __GMP_NOTHROW __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_pow_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);

/* undocumented now, but retained here for upward compatibility */
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_preinv_mod_1(mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_random(mp_ptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_random2(mp_ptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_rshift(mp_ptr, mp_srcptr, mp_size_t, unsigned int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_scan0(mp_srcptr, mp_bitcnt_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_bitcnt_t gpmpn_scan1(mp_srcptr, mp_bitcnt_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_set_str(mp_ptr, const unsigned char *, size_t, int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE size_t gpmpn_sizeinbase(mp_srcptr, mp_size_t, int);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sqrtrem(mp_ptr, mp_ptr, mp_srcptr, mp_size_t);

#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_sub)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#endif

#if __GMP_INLINE_PROTOTYPES || defined(__GMP_FORCE_gpmpn_sub_1)
//__GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t) __GMP_NOTHROW;
#endif

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sub_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_submul_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_tdiv_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t*);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_tdiv_qr_itch(mp_size_t, mp_size_t);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_tdiv_qr_itch(mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_and_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_andn_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_nand_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_ior_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_iorn_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_nior_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_xor_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_xnor_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_copyi(mp_ptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_copyd(mp_ptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_zero(mp_ptr, mp_size_t);


        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_cnd_add_n(mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_cnd_sub_n(mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sec_add_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_add_1_itch(mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sec_sub_1(mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_sub_1_itch(mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_cnd_swap(mp_limb_t, volatile mp_limb_t *, volatile mp_limb_t *, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_mul(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_mul_itch(mp_size_t, mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_sqr(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_sqr_itch(mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_powm(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_bitcnt_t, mp_srcptr, mp_size_t, mp_ptr);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_powm_itch(mp_size_t, mp_bitcnt_t, mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_tabselect(volatile mp_limb_t *, volatile const mp_limb_t *, mp_size_t, mp_size_t, mp_size_t);

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_limb_t gpmpn_sec_div_qr(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_div_qr_itch(mp_size_t, mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpmpn_sec_div_r(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE mp_size_t gpmpn_sec_div_r_itch(mp_size_t, mp_size_t) __GMP_ATTRIBUTE_PURE;

        __GPGMP_DECLSPEC __GPGMP_CALLERTYPE int gpmpn_sec_invert(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_bitcnt_t, mp_ptr);
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
            __GPGMPN_COPY_REST(wp, xp, xsize, __gmp_i);                \
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
#define __GPGMPN_AORS_1(cout, dst, src, n, v, OP, CB)             \
    do                                                            \
    {                                                             \
        mp_size_t __gmp_i;                                        \
        mp_limb_t __gmp_x, __gmp_r;                               \
                                                                  \
        /* ASSERT ((n) >= 1); */                                  \
        /* ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, n)); */      \
                                                                  \
        __gmp_x = (src)[0];                                       \
        __gmp_r = __gmp_x OP(v);                                  \
        (dst)[0] = __gmp_r;                                       \
        if (CB(__gmp_r, __gmp_x, (v)))                            \
        {                                                         \
            (cout) = 1;                                           \
            for (__gmp_i = 1; __gmp_i < (n);)                     \
            {                                                     \
                __gmp_x = (src)[__gmp_i];                         \
                __gmp_r = __gmp_x OP 1;                           \
                (dst)[__gmp_i] = __gmp_r;                         \
                ++__gmp_i;                                        \
                if (!CB(__gmp_r, __gmp_x, 1))                     \
                {                                                 \
                    if ((src) != (dst))                           \
                        __GPGMPN_COPY_REST(dst, src, n, __gmp_i); \
                    (cout) = 0;                                   \
                    break;                                        \
                }                                                 \
            }                                                     \
        }                                                         \
        else                                                      \
        {                                                         \
            if ((src) != (dst))                                   \
                __GPGMPN_COPY_REST(dst, src, n, 1);               \
            (cout) = 0;                                           \
        }                                                         \
    } while (0)
#endif

#if GMP_NAIL_BITS >= 1
#define __GPGMPN_AORS_1(cout, dst, src, n, v, OP, CB)             \
    do                                                            \
    {                                                             \
        mp_size_t __gmp_i;                                        \
        mp_limb_t __gmp_x, __gmp_r;                               \
                                                                  \
        /* ASSERT ((n) >= 1); */                                  \
        /* ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, n)); */      \
                                                                  \
        __gmp_x = (src)[0];                                       \
        __gmp_r = __gmp_x OP(v);                                  \
        (dst)[0] = __gmp_r & GMP_NUMB_MASK;                       \
        if (__gmp_r >> GMP_NUMB_BITS != 0)                        \
        {                                                         \
            (cout) = 1;                                           \
            for (__gmp_i = 1; __gmp_i < (n);)                     \
            {                                                     \
                __gmp_x = (src)[__gmp_i];                         \
                __gmp_r = __gmp_x OP 1;                           \
                (dst)[__gmp_i] = __gmp_r & GMP_NUMB_MASK;         \
                ++__gmp_i;                                        \
                if (__gmp_r >> GMP_NUMB_BITS == 0)                \
                {                                                 \
                    if ((src) != (dst))                           \
                        __GPGMPN_COPY_REST(dst, src, n, __gmp_i); \
                    (cout) = 0;                                   \
                    break;                                        \
                }                                                 \
            }                                                     \
        }                                                         \
        else                                                      \
        {                                                         \
            if ((src) != (dst))                                   \
                __GPGMPN_COPY_REST(dst, src, n, 1);               \
            (cout) = 0;                                           \
        }                                                         \
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

#if defined(__GPGMPN_COPY) && !defined(__GPGMPN_COPY_REST)
#define __GPGMPN_COPY_REST(dst, src, size, start)                          \
    do                                                                     \
    {                                                                      \
        /* ASSERT ((start) >= 0); */                                       \
        /* ASSERT ((start) <= (size)); */                                  \
        __GPGMPN_COPY((dst) + (start), (src) + (start), (size) - (start)); \
    } while (0)
#endif

/* Copy {src,size} to {dst,size}, starting at "start".  This is designed to
   keep the indexing dst[j] and src[j] nice and simple for __GMPN_ADD_1,
   __GMPN_ADD, etc.  */
#if !defined(__GPGMPN_COPY_REST)
#define __GPGMPN_COPY_REST(dst, src, size, start)               \
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

#if !defined(__GPGMPN_COPY)
#define __GPGMPN_COPY(dst, src, size) __GPGMPN_COPY_REST(dst, src, size, 0)
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_add)
#if !defined(__GMP_FORCE_gpmpn_add)
        __GMP_EXTERN_INLINE
#endif
        ANYCALLER static inline mp_limb_t
        gpmpn_add(mp_ptr __gmp_wp, mp_srcptr __gmp_xp, mp_size_t __gmp_xsize, mp_srcptr __gmp_yp, mp_size_t __gmp_ysize)
        {
            mp_limb_t __gmp_c;
            __GPGMPN_ADD(__gmp_c, __gmp_wp, __gmp_xp, __gmp_xsize, __gmp_yp, __gmp_ysize);
            return __gmp_c;
        }
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_add_1)
#if !defined(__GMP_FORCE_gpmpn_add_1)
        __GMP_EXTERN_INLINE
#endif
        ANYCALLER static inline mp_limb_t
        gpmpn_add_1(mp_ptr __gmp_dst, mp_srcptr __gmp_src, mp_size_t __gmp_size, mp_limb_t __gmp_n) __GMP_NOTHROW
        {
            mp_limb_t __gmp_c;
            __GPGMPN_ADD_1(__gmp_c, __gmp_dst, __gmp_src, __gmp_size, __gmp_n);
            return __gmp_c;
        }
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_cmp)
#if !defined(__GMP_FORCE_gpmpn_cmp)
        __GMP_EXTERN_INLINE
#endif
        ANYCALLER static inline int
        gpmpn_cmp(mp_srcptr __gmp_xp, mp_srcptr __gmp_yp, mp_size_t __gmp_size) __GMP_NOTHROW
        {
            int __gmp_result;
            __GPGMPN_CMP(__gmp_result, __gmp_xp, __gmp_yp, __gmp_size);
            return __gmp_result;
        }
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_zero_p)
#if !defined(__GMP_FORCE_gpmpn_zero_p)
        __GMP_EXTERN_INLINE
#endif
        ANYCALLER static inline int gpmpn_zero_p(mp_srcptr __gmp_p, mp_size_t __gmp_n) __GMP_NOTHROW
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
        ANYCALLER static inline mp_limb_t
        gpmpn_sub(mp_ptr __gmp_wp, mp_srcptr __gmp_xp, mp_size_t __gmp_xsize, mp_srcptr __gmp_yp, mp_size_t __gmp_ysize)
        {
            mp_limb_t __gmp_c;
            __GPGMPN_SUB(__gmp_c, __gmp_wp, __gmp_xp, __gmp_xsize, __gmp_yp, __gmp_ysize);
            return __gmp_c;
        }
#endif

#if defined(__GMP_EXTERN_INLINE) || defined(__GMP_FORCE_gpmpn_sub_1)
#if !defined(__GMP_FORCE_gpmpn_sub_1)
        __GMP_EXTERN_INLINE
#endif
        ANYCALLER static inline mp_limb_t
        gpmpn_sub_1(mp_ptr __gmp_dst, mp_srcptr __gmp_src, mp_size_t __gmp_size, mp_limb_t __gmp_n) __GMP_NOTHROW
        {
            mp_limb_t __gmp_c;
            __GPGMPN_SUB_1(__gmp_c, __gmp_dst, __gmp_src, __gmp_size, __gmp_n);
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

ANYCALLER void perform_udiv_qr_3by2(mp_limb_t& a, mp_limb_t& b, mp_limb_t& c, mp_limb_t& d, mp_limb_t& e, mp_limb_t f, mp_limb_t& g, mp_limb_t& h, mp_limb_t& i);

ANYCALLER mp_limb_t perform_gpmpn_submul_1(mp_ptr a, mp_srcptr b, mp_size_t c, mp_limb_t d);

ANYCALLER mp_limb_t perform_gpmpn_add_n(mp_ptr a, mp_srcptr b, mp_srcptr c, mp_size_t d);

ANYCALLER void perform_MPN_COPY(mp_ptr qp, mp_srcptr tp, mp_size_t qn);

#define mpf_array_sgn(mpfArrayIdxStruct) (MPF_ARRAY_SIZES(mpfArrayIdxStruct.array)[mpfArrayIdxStruct.idx] < 0 ? -1 : MPF_ARRAY_SIZES(mpfArrayIdxStruct.array)[mpfArrayIdxStruct.idx] > 0)

    }
}