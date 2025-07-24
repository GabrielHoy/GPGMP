#pragma once
#include "GPGMP/gpgmp.cuh"

namespace gpgmp {

    namespace host {

        //Initializes a new mpf_array struct, this is used after a struct has been allocated in order to zero its memory out.
        //?Optimization potential: come back and just use one memset with the entire memory block since it's contiguous, not as pretty or easy to debug though
        HOSTONLY static inline void mpf_array_init(const mpf_host_array array) {
            //Go through each size in the array and set it to 0, since every number in the array starts out utilizing 0 limbs.
            memset(MPF_ARRAY_SIZES(array), 0, array->numFloatsInArray * sizeof(int));
            //Go through each exponent in the array and set it to 0, since every number in the array starts out with an exponent of 0.
            memset(MPF_ARRAY_EXPONENTS(array), 0, array->numFloatsInArray * sizeof(mp_exp_t));
            //Finally, go through each limb in the actual array's data and set it to 0 as well.
            memset(MPF_ARRAY_DATA(array), 0, array->numFloatsInArray * array->limbsPerArrayFloat * sizeof(mp_limb_t));
        }

        //Initializes a single integer inside of an mpf_array struct to zero.
        HOSTONLY static inline void mpf_array_init_idx(const mpf_host_array array, const int idx) {
            //Exponent goes to 0 since we're initializing to 0...
            MPF_ARRAY_EXPONENTS(array)[idx] = 0;
            //Then set the size of this float to 0, since we're initializing to 0...
            MPF_ARRAY_SIZES(array)[idx] = 0;
            //Then we go through each limb that this float's data occupies and....we set its value to 0!!!
            memset(MPF_ARRAY_DATA(array) + (idx * array->limbsPerArrayFloat), 0, array->limbsPerArrayFloat * sizeof(mp_limb_t));
        }

        //Initializes an mpf_array struct by copying the values of a default gmp mpf_t array into it.
        //This is common when you're working with "normal" GMP mpf_t[n]'s and you want to convert them over to mpf_array's for GPU use
        HOSTONLY static inline void mpf_array_init_from_gmp_array_of_mpf(const mpf_host_array array, const mpf_t* gmpMpfArray, const int gmpMpfArraySize) {
            int* sizesArray = MPF_ARRAY_SIZES(array);
            mp_exp_t* exponentsArray = MPF_ARRAY_EXPONENTS(array);
            mp_limb_t* dataArray = MPF_ARRAY_DATA(array);

            //Go through each float in the gmp mpf_t array and copy its values over to the mpf_array struct.
            for (int i = 0; i < gmpMpfArraySize; i++) {
                //Get the current mpf_t...
                const mpf_t& currentMpf = gmpMpfArray[i];
                ASSERT(ABSIZ(currentMpf) <= array->limbsPerArrayFloat-1); //Ensure that the current mpf_t actually fits inside of the mpf_array we're copying over to...
                //copy over the exponent and size of the current mpf_t into the mpf_array struct's data
                exponentsArray[i] = EXP(currentMpf);
                sizesArray[i] = SIZ(currentMpf);
                //finally, memcpy over the limb data
                memcpy(dataArray + (i * array->limbsPerArrayFloat), PTR(currentMpf), ABSIZ(currentMpf) * sizeof(mp_limb_t));
            }
        }

        //Initializes an mpf_array struct on the device from the host, with an optional gpuStream_t for synchronization.
        //Requires parameters to be passed describing the array size as well as the bits used for precision, this is due to the deviceArrayPtr's data being inaccessible from the host.
        //Returns a cudaError_t error code associated with the memset operation.
        //
        // **NOTE: From experimentation, it seems like this may not be required for device arrays if you simply want to initialize them to 0 -- CUDA allocations seem to zero out memory automatically. Still writing this function for robustness regardless...
        HOSTONLY static inline cudaError_t mpf_array_init_on_device_from_host(const mpf_device_array deviceArrayPtr, const int numElementsInArray, const mp_bitcnt_t precisionPerFloat, cudaStream_t stream = 0) {
            if (stream) {
                //Zero out all of the data in the 'sizes array', 'exponents array', and 'data array' memory following the deviceArrayPtr's struct.
                //These are in one contiguous memory block, so we only need a single memset call to achieve what we want, yay!
                return cudaMemsetAsync(
                    MPF_ARRAY_SIZES_NO_PTR_INDEXING(deviceArrayPtr),
                    0, //...setting the memory to 0's...
                    ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * numElementsInArray) //Byte count to set to 0's...Size of the sizes array after the struct,
                        + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * numElementsInArray) //...and the exponents array after the sizes array,
                        + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * (LIMB_COUNT_FROM_PRECISION_BITS(precisionPerFloat)+1) * numElementsInArray), //...and finally the proper "data" array after the exponents array.
                    stream //...and we're using the given stream for synchronization.
                );
            }
            else {
                //Same deal as the above branch, simply without a stream.
                return cudaMemset(
                    MPF_ARRAY_SIZES_NO_PTR_INDEXING(deviceArrayPtr),
                    0, //...setting the memory to 0's...
                    ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * numElementsInArray)
                        + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_exp_t) * numElementsInArray)
                        + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * (LIMB_COUNT_FROM_PRECISION_BITS(precisionPerFloat)+1) * numElementsInArray)
                );
            }
        }

        //Initializes an mpf_array struct on the device from the host, copying over the values of a "gmp style" mpf_t array into the mpf_array struct - optionally using a gpuStream_t for synchronization.
        //Requires parameters to be passed describing the device array size as well as the bits used for precision, this is due to the deviceArrayPtr's data being inaccessible from the host.
        //Returns a cudaError_t error code associated with ANY memcpy operation that may fail when copying over said values.
        // **OPTIMIZATION FOR LATER** - I should 1,000% be using cudaMemcpyBatchAsync and cudaMemcpyBatch. I am not for now for the sake of simplicity and getting a working product. I realize that this is not a good excuse, and that this is a significant source of performance loss.
        HOSTONLY static inline cudaError_t mpf_array_init_on_device_from_gmp_array_of_mpf(mpf_device_array deviceArrayPtr, const mpf_t* gmpMpfArray, const int numElementsInDeviceArray, const mp_bitcnt_t precisionPerFloat, const int numElementsInGmpMpfArray, cudaStream_t stream = 0) {
            int* deviceSizesArrayPtr = MPF_ARRAY_SIZES_NO_PTR_INDEXING(deviceArrayPtr);
            size_t limbsPerFloat = LIMB_COUNT_FROM_PRECISION_BITS(precisionPerFloat)+1;
            mp_exp_t* deviceExponentsArrayPtr = MPF_ARRAY_EXPONENTS_NO_PTR_INDEXING(deviceArrayPtr, numElementsInDeviceArray);
            mp_limb_t* deviceDataArrayPtr = MPF_ARRAY_DATA_NO_PTR_INDEXING(deviceArrayPtr, numElementsInDeviceArray);

            cudaError_t err;

            if (stream) {
                //Go through each float in the mpf_t array and initialize it in our mpf_array struct.
                for (int i = 0; i < numElementsInGmpMpfArray; i++) {
                    //Get the current mpf_t float.
                    const mpf_t& mpfToCopy = gmpMpfArray[i];
                    ASSERT(ABSIZ(mpfToCopy) <= (limbsPerFloat-1)); //Ensure that the current mpf_t object actually fits inside of the mpf_array...
                    //Copying over the size of the current mpf_t object
                    err = cudaMemcpyAsync(
                        deviceSizesArrayPtr + i,
                        &mpfToCopy->_mp_size,
                        sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }

                    //...Then copying over the exponent of the object
                    err = cudaMemcpyAsync(
                        deviceExponentsArrayPtr + i,
                        &mpfToCopy->_mp_exp,
                        sizeof(mp_exp_t),
                        cudaMemcpyHostToDevice,
                        stream
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }

                    //...Finally, copying over the actual limb data
                    err = cudaMemcpyAsync(
                        deviceDataArrayPtr + (i * limbsPerFloat),
                        PTR(mpfToCopy),
                        ABSIZ(mpfToCopy) * sizeof(mp_limb_t),
                        cudaMemcpyHostToDevice,
                        stream
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }
                }
            }
            else {
                //Same deal as the above branch, but without a stream.
                for (int i = 0; i < numElementsInGmpMpfArray; i++) {
                    const mpf_t& mpfToCopy = gmpMpfArray[i];
                    ASSERT(ABSIZ(mpfToCopy) <= (limbsPerFloat-1));
                    err = cudaMemcpy(
                        deviceSizesArrayPtr + i,
                        &mpfToCopy->_mp_size,
                        sizeof(int),
                        cudaMemcpyHostToDevice
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }

                    err = cudaMemcpy(
                        deviceExponentsArrayPtr + i,
                        &mpfToCopy->_mp_exp,
                        sizeof(mp_exp_t),
                        cudaMemcpyHostToDevice
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }

                    err = cudaMemcpy(
                        deviceDataArrayPtr + (i * limbsPerFloat),
                        PTR(mpfToCopy),
                        ABSIZ(mpfToCopy) * sizeof(mp_limb_t),
                        cudaMemcpyHostToDevice
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }
                }
            }

            return err;
        }

    }


    namespace device {

        //Initializes a single integer inside of an mpf_array struct to zero.
        GPUONLY static inline void mpf_array_init_idx(const mpf_device_array array, const int idx) {
            //Set the size & exponent of this float to 0 limbs since its value is now 0.
            MPF_ARRAY_SIZES(array)[idx] = 0;
            MPF_ARRAY_EXPONENTS(array)[idx] = 0;
            //Then go through each limb that this float's data occupies and set its value to 0.
            mp_limb_t* dataArray = MPF_ARRAY_DATA(array);
            for (int i = idx * array->limbsPerArrayFloat; i < (idx + 1) * array->limbsPerArrayFloat; i++) {
                dataArray[i] = 0;
            }
        }

        //Initializes a new mpf_array struct on the current CUDA device, this is used after a struct has been allocated in order to zero its memory out.
        //This is VERY SLOW due to the fact that it has to zero out every limb in the array, and if I remember correctly memset on CUDA kernels forces serial execution.
        //It is heavily recommended to do this kind of initialization on the host side instead of inside a kernel.
        GPUONLY static inline void mpf_array_init_SLOW(const mpf_device_array array) {
            //Go through each size in the array and set it to 0, since every number in the array is now 0 limbs long.
            memset(MPF_ARRAY_SIZES(array), 0, array->numFloatsInArray * sizeof(int));
            //...Go through each exponent in the array and set it to 0...since every number in the array is now 0 limbs long...
            memset(MPF_ARRAY_EXPONENTS(array), 0, array->numFloatsInArray * sizeof(mp_exp_t));
            //...Finally, go through each limb in the array and set its data to 0 as well.
            memset(MPF_ARRAY_DATA(array), 0, array->numFloatsInArray * array->limbsPerArrayFloat * sizeof(mp_limb_t));
        }

    }

}