/*
  GPGMP
  Gabriel Hoy, 2025
  A wrapper around the GMP library and several of its routines, designed to enable GPU-accelerated batch operations on Arbitrary Precision numbers.


  This library is designed to be used in conjunction with GMP for maximum performance, NOT as a full replacement of GMP.
  It is important to note that the serial execution speed of GPGMP operations will not match that of GMP operations, the intended performance improvement from using GPGMP is designed to come from the parallelization of operations on mass amounts of numbers.

  GPU execution of Multiple-Precision Numbers is achieved by:
  - Copying arrays of GMP numbers into gpgmp::mpX_array's that can be either sent to or initialized on the GPU
  - Performing the desired operations on the mpX_arrays in your GPU kernels
  - Optionally copying the results back into GMP numbers(if gpgmp does not support your desired functionality)
*/

#pragma once


//A version of the gmp.h header comes included with the gpgmp library, though you should also link with gmp.
//An attempt is made to support differences in gmp versions, but this is by NO means guarunteed and should NOT be expected to operate correctly.
//In particular, there is zero support currently attempted to facilitate GMP_NAIL_BITS != 0.
//If you're having trouble with this library, it's possible that the version of gmp.h you're using is not supported. Try using the version that comes with gpgmp if you can.
#ifndef __GMP_H__
#include "gmp.cuh"
#endif

#include "Definitions.cuh"

//Quick reference of all available namespaces in the gpgmp library, for anyone who may be reading this code:
namespace gpgmp { //Shared functions callable between both host and device are placed in the root gpgmp namespace.
  namespace host {} //Host-only functions
  namespace device {} //Device-only functions
  namespace internal {} //Internal functions primarily meant for use within the GPGMP library itself, and not particularly for use/useful by end users.

  //Functions shared between both host and device for performing operations on GMP's integer numbers - the mpn_t struct.
  //This is (an attempt) at a port of all of the GMP "mpn_X" routines to be runnable on the GPU.
  //Syntax should be equivalent to the GMP library.
  namespace mpnRoutines {}

  //Functions shared between both host and device for performing operations on GMP's floating-point numbers - the mpf_t struct.
  //This is (an attempt) at a port of all of the GMP "mpf_X" routines to be runnable on the GPU.
  //
  //Syntax is NOT equivalent to the GMP library! Functions have equivalent names, however a lot of them now take an additional mp_ptr "scratchSpace"
  //if you run into this(you almost certainly will) you can use the gpgmp::internal::mpf_get_scratch_space_limb_count function with a bit-field of operations you want to perform, this function will spit out a number of mp_limb_t's that need to be allocated in order to provide any of the given operations - you can then pass a pointer to those limbs into the function.
  //Previously the functions simply called malloc at call-time in order to allocate necessary scratch space and not bother the user with having to provide it, but to be usable on the GPU we need to move all dynamic allocation to be preemptively done by the host, so this requirement now falls onto the user.
  namespace mpfRoutines {}

  //Functions shared between both host and device for performing operations on GPGMP's GPU-optimized arrays of floating-point numbers - the mpf_array struct.
  //The functions in this namespace will follow similar naming conventions to the mpfRoutines namespace, but are designed to operate on the formatting and structure of the mpf_array struct.
  //Note that these functions are NOT cross-compatible with GMP's mpf_t structs!
  namespace mpfArrayRoutines {}
}

namespace gpgmp {
  /*

    Original GMP mpz_t struct for reference:
    typedef struct
    {
      int _mp_alloc;		Number of *limbs* allocated and pointed
              to by the _mp_d field.
      int _mp_size;			abs(_mp_size) is the number of limbs the
              last field points to.  If _mp_size is
              negative this is a negative number.
      mp_limb_t *_mp_d;		Pointer to the limbs.
    } __mpz_struct;
  */

  /*
    Struct used to store an array of mpz_t's in a manner that allows for better GPU memory coalescence.
    This struct is NOT designed to be created directly, instead use mpn_array_allocate_X() to retrieve one!
    Every time this struct is allocated, space for its arrays is also allocated directly after the struct.

    !IMPORTANT: mpn_array's behave differently than GMP's mpz_t structs! They do not dynamically reallocate memory to contain bigger numbers.
    This means that you must make sure you have enough precision allocated to hold whatever numbers you're going to end up storing.

    You can access the limbs for a particular number in the array via array->operator[](idx); however you CANNOT treat the standalone "array" as index 0 - you need to explicitly index the 0th element!
    This is because the [] operator is actually indexing the memory directly after the mpn_array struct, where the limbs for the actual numbers are stored -- it also handles offsetting based on limb count per number accordingly,
    so all you need to do to get number X is array->operator[](x) instead of dealing with limb sizes etc etc... This still isn't pretty but I'm not too sure how to improve it and this works.
  */
  struct __align__(128) mpn_array {
      int numIntegersInArray; //Number of integers that this array contains.
      int numLimbsPerInteger; //Number of limbs allocated for each integer in the array to use(not all of these limbs will likely be used, this is a maximum.)

      //If you're familiar with GMP, you may ask 'where is the mp_d and mp_size data?'
      //The array data and sizes for numbers in this array are stored directly after the struct as a contiguous block of memory.
      //The MPN_ARRAY_DATA and MPN_ARRAY_SIZES macros can be used to easily access these arrays.
      //It would be cleaner to define mp_limb_t* _mp_array_data and int* _mp_sizes in this struct and have them point to other dynamically allocated arrays, but that would be less performant.

    //Helper operators for array indexing so that users can use common syntax like arr[0], arr[1], arr[2]...
    ANYCALLER mp_limb_t& operator[](int index) {
      return *(MPN_ARRAY_DATA(this) + (index * this->numLimbsPerInteger));
    }

    ANYCALLER const mp_limb_t& operator[](int index) const {
      return *(MPN_ARRAY_DATA_CONST(this) + (index * this->numLimbsPerInteger));
    }
  };

  //This is a direct alias for an mpn_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpn_array is on the host or the device.
  typedef mpn_array* mpn_device_array;
  //This is a direct alias for an mpn_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpn_array is on the host or the device.
  typedef mpn_array* mpn_host_array;


  /*

  Original GMP mpf struct for reference:
  typedef struct
  {
    int _mp_prec;			// Max precision, in number of `mp_limb_t's.
            Set by mpf_init and modified by
            mpf_set_prec.  The area pointed to by the
            _mp_d field contains `prec' + 1 limbs.
    int _mp_size;			// abs(_mp_size) is the number of limbs the
            last field points to.  If _mp_size is
            negative this is a negative number.
    mp_exp_t _mp_exp;		// Exponent, in the base of `mp_limb_t'.
    mp_limb_t *_mp_d;		// Pointer to the limbs.
  } __mpf_struct;
  */

  //Flags for specifying which operations a user wants to be available for use on a given floating-point number.
  //These flags specify the amount of scratch space required for a floating-point number, used to pre-allocate this space when allocating a gpmpf_t/mpf_array.
  //The amount of scratch space doesn't stack between operations - the total allocated extra space is determined by the maximum requirement among the given operation flags.
  enum UsedOperationFlags {
    OP_NONE = 0, //Ignore...
    OP_ADD = 1 << 0, //Addition between two floating-point numbers.
    OP_SUB = 1 << 1, //Subtraction between two floating-point numbers.
    OP_MUL = 1 << 2, //Multiplication between two floating-point numbers.
    OP_DIV = 1 << 3, //Division between two floating-point numbers.
    OP_UI_DIV = 1 << 4, //Division of unsigned integers by floating-point numbers.
    OP_DIV_UI = 1 << 5, //Division of floating-point numbers by unsigned integers.
    OP_SQRT = 1 << 6, //Square root of a floating-point number.
    OP_SQRT_UI = 1 << 7, //Square root of an unsigned integer.
    OP_RELDIFF = 1 << 8 //Absolute difference between two floating-point numbers.
  };
  typedef uint32_t _UsedOperationFlags; //A direct alias for uint32_t, used to identify the enum type used inside of available operation bit-fields in a more readable manner.

  //A struct that stores an array of gpmpf_t's in a manner that allows for better GPU memory coalescence.
  struct __align__(128) mpf_array {
    int numFloatsInArray; //Number of floating-point numbers that this array contains.
    int userSpecifiedPrecisionLimbCount; //Number of limbs that the user specified for the precision of each floating-point number in this array.
    int limbsPerArrayFloat; //Number of limbs *actually allocated* for each floating-point number in this array to use - this accounts for any "scratch space" limbs.
    _UsedOperationFlags availableOperations; //Bitfield of UsedOperationFlags - Flags specifying which operations are available for use on a given floating-point number.

    //...Also contains arrays immediately following the struct similar to mpn_array - these arrays are:
    //- "_mp_sizes" - an array of SIGNED ints, representing how many limbs are used by each float in the array.
    //- "_mp_exponents" - an array of mp_exp_t's(usually typedefed long int's) - representing the exponent of each float in the array.
    //- "_mp_data" - an array - of arrays - of mp_limb_t's, flattened out into 1D - representing the total limb data for each float in the array. i.e an mpf_array[2] with a precision of 2 limbs could have a data array that reads {0,42,1,0}, thereby arr[0] = ((2^64)*0) + 42 and arr[1] = ((2^64)*1) + 0
    //
    //Important notes about the "_mp_data" array that differs from GMP behavior:
    // Usually GMP "lies" about the number of limbs that are allocated by a float and adds one to the actual desired precision from the user; they however still store the desired, lower precision value inside of _mp_prec.
    // We replicate most of this behavior, but we more directly expose the actual precision that the float "has" by storing the "value+1" count of limbs inside limbsPerArrayFloat, instead of storing the user-specified number and just allocating n+1 limbs.
    // This means that when implementing any formulas you can simply use limbsPerArrayFloat as the source-of-truth for stride values etc, without needing to worry about this "trailing" limb interfering with any of your pointer offsets, etc.


    //Helper operators for array indexing so that users can use common syntax like arr[0], arr[1], arr[2]...
    ANYCALLER mp_limb_t& operator[](int index) {
      return *(MPF_ARRAY_DATA(this) + (index * this->limbsPerArrayFloat));
    }

    ANYCALLER const mp_limb_t& operator[](int index) const {
      return *(MPF_ARRAY_DATA_CONST(this) + (index * this->limbsPerArrayFloat));
    }
  };

  //This is a direct alias for an mpf_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpf_array is on the host or the device.
  typedef mpf_array* mpf_device_array;
  //This is a direct alias for an mpf_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpf_array is on the host or the device.
  typedef mpf_array* mpf_host_array;
}

#include "mpn/mpn.cuh"
#include "mpf/mpf.cuh"