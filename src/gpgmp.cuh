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
#include "gmp.h"
#endif

#include "Definitions.cuh"

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

  //Struct used to store an array of mpz_t's in a manner that allows for better GPU memory coalescence.
  //This struct is NOT designed to be used directly, instead use mpn_array_allocate_X() to retrieve one!
  //Every time this struct is allocated, space for its arrays is also allocated directly after the struct.
  //!IMPORTANT: mpn_array's behave differently than GMP's mpz_t structs! They do not dynamically reallocate memory to contain bigger numbers.
  //This means that you must make sure you have enough precision allocated to hold whatever numbers you're going to end up storing.
  struct __align__(128) mpn_array {
      int numIntegersInArray; //Number of integers that this array contains.

      int numLimbsPerInteger; //Number of limbs allocated for each integer in the array to use(not all of these limbs will likely be used, this is a maximum.)

      //!IMPORTANT: Where are the 'array data' and 'integer sizes' arrays?
      //The array data and sizes for this array are stored directly after the struct as a contiguous block of memory.
      //The MPN_ARRAY_DATA and MPN_ARRAY_SIZES macros can be used to easily access these arrays.
      //It would be cleaner to define mp_limb_t* _mp_array_data and int* _mp_sizes in this struct and have them point to other dynamically allocated arrays, but that would be less performant.
  };

  //This is a direct alias for an mpn_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpn_array is on the host or the device.
  typedef mpn_array* mpn_device_array;
  //This is a direct alias for an mpn_array pointer; it is used to facilitate better type clarity within user code so that confusion does not arise between whether an mpn_array is on the host or the device.
  typedef mpn_array* mpn_host_array;

}
#include "mpn/mpn.cuh"