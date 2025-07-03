#pragma once
#include "gpgmp-impl.cuh"

namespace gpgmp {

    namespace host {
        //Initializes a new mpn_array struct, this is used after a struct has been allocated in order to zero its memory out.
        HOSTONLY void mpn_array_init(const mpn_host_array array) {
            //Go through each limb in the array and set it to 0.
            memset(MPN_ARRAY_DATA(array), 0, array->numIntegersInArray * array->numLimbsPerInteger * sizeof(mp_limb_t));
            //Go through each size in the array and set it to 0, since every number in the array is now 0 limbs long.
            memset(MPN_ARRAY_SIZES(array), 0, array->numIntegersInArray * sizeof(int));
        }

        //Initializes a single integer inside of an mpn_array struct to zero.
        HOSTONLY void mpn_array_init_idx(const mpn_host_array array, const int idx) {
            //Go through each limb that this integer occupies and set its value to 0.
            memset(MPN_ARRAY_DATA(array) + (idx * array->numLimbsPerInteger), 0, array->numLimbsPerInteger * sizeof(mp_limb_t));

            //Then set the size of this integer to 0 limbs since its value is now 0.
            MPN_ARRAY_SIZES(array)[idx] = 0;
        }

        //Initializes an mpn_array struct by copying the values of an mpz_t array into it.
        HOSTONLY void mpn_array_init_from_mpz_array(const mpn_host_array array, const mpz_t* mpzArray, const int mpzArraySize) {
            mp_limb_t* dataArray = MPN_ARRAY_DATA(array);
            int* sizesArray = MPN_ARRAY_SIZES(array);

            //Go through each integer in the mpz_t array and initialize it in the mpn_array struct.
            for (int i = 0; i < mpzArraySize; i++) {
                //Get the current mpz_t integer.
                const mpz_t& mpzToCopy = mpzArray[i];
                ASSERT(ABSIZ(mpzToCopy) <= array->numLimbsPerInteger); //Ensure that the current mpz_t integer actually fits inside of the mpn_array.
                memcpy(dataArray + (i * array->numLimbsPerInteger), PTR(mpzToCopy), ABSIZ(mpzToCopy) * sizeof(mp_limb_t));
                sizesArray[i] = SIZ(mpzToCopy);
            }
        }

        //Initializes an mpn_array struct on the device from the host, with an optional gpuStream_t for synchronization.
        //Requires parameters to be passed describing the array size as well as the bits used for precision, this is due to the deviceArrayPtr's data being inaccessible from the host.
        //Returns a cudaError_t error code associated with the memset operation.
        //
        // **NOTE: From experimentation, it seems like this may not be required for device arrays if you simply want to initialize them to 0 -- CUDA allocations seem to zero out memory automatically. (All cards on the table, I don't know how true this is. I am a beginner CUDA developer.)
        HOSTONLY cudaError_t mpn_array_init_on_device_from_host(const mpn_device_array deviceArrayPtr, const int arraySize, const mp_bitcnt_t precision, cudaStream_t stream = 0) {
            if (stream) {
                //Zero out all of the data in the 'data array' and 'sizes array' memory following the deviceArrayPtr's struct.
                //These are in one contiguous memory block, so we only need a single memset call to achieve what we want, yay!
                return cudaMemsetAsync(
                    MPN_ARRAY_DATA_NO_PTR_INDEXING(deviceArrayPtr),
                    0, //...setting the memory to 0's...
                    ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * LIMB_COUNT_FROM_PRECISION_BITS(precision) * arraySize) //Byte count to set to 0's...Size of the data array after the struct
                        + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * arraySize), //...and the "sizes" array after the data array, too.
                    stream //...and we're using the given stream for synchronization.
                );
            }
            else {
                //Same deal as the above branch, simply without a stream.
                return cudaMemset(
                    MPN_ARRAY_DATA_NO_PTR_INDEXING(deviceArrayPtr),
                    0, //...setting the memory to 0's...
                    ALIGN_TO_128_BYTE_MULTIPLE(sizeof(mp_limb_t) * LIMB_COUNT_FROM_PRECISION_BITS(precision) * arraySize)
                        + ALIGN_TO_128_BYTE_MULTIPLE(sizeof(int) * arraySize)
                );
            }
        }

        //Initializes an mpn_array struct on the device from the host, copying over the values of an mpz_t array into the mpn_array struct - optionally using a gpuStream_t for synchronization.
        //Requires parameters to be passed describing the device array size as well as the bits used for precision, this is due to the deviceArrayPtr's data being inaccessible from the host.
        //Returns a cudaError_t error code associated with ANY memcpy operation that may fail when copying over said values.
        // **OPTIMIZATION FOR LATER** - I should 1,000% be using cudaMemcpyBatchAsync and cudaMemcpyBatch. I am not for now for the sake of simplicity and getting a working product. I realize that this is not a good excuse, and that this is a significant source of performance loss.
        HOSTONLY cudaError_t mpn_array_init_on_device_from_mpz_array(mpn_device_array deviceArrayPtr, const mpz_t* mpzArray, const int arraySize, const mp_bitcnt_t precision, const int mpzArraySize, cudaStream_t stream = 0) {
            mp_limb_t* deviceDataArrayPtr = MPN_ARRAY_DATA_NO_PTR_INDEXING(deviceArrayPtr);
            size_t limbsPerInteger = LIMB_COUNT_FROM_PRECISION_BITS(precision); //equivalent to deviceArrayPtr->numLimbsPerInteger...
            int* deviceSizesArrayPtr = MPN_ARRAY_SIZES_NO_PTR_INDEXING(deviceArrayPtr, arraySize, precision);

            cudaError_t err;

            if (stream) {
                //Go through each integer in the mpz_t array and initialize it in the mpn_array struct.
                for (int i = 0; i < mpzArraySize; i++) {
                    //Get the current mpz_t integer.
                    const mpz_t& mpzToCopy = mpzArray[i];
                    ASSERT(ABSIZ(mpzToCopy) <= limbsPerInteger); //Ensure that the current mpz_t integer actually fits inside of the mpn_array.
                    err = cudaMemcpyAsync(
                        deviceDataArrayPtr + (i * limbsPerInteger),
                        mpzToCopy->_mp_d,
                        ABSIZ(mpzToCopy) * sizeof(mp_limb_t),
                        cudaMemcpyHostToDevice,
                        stream
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }

                    err = cudaMemcpyAsync(
                        deviceSizesArrayPtr + i,
                        &SIZ(mpzToCopy),
                        sizeof(int),
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
                for (int i = 0; i < mpzArraySize; i++) {
                    //Get the current mpz_t integer.
                    const mpz_t& mpzToCopy = mpzArray[i];
                    ASSERT(ABSIZ(mpzToCopy) <= limbsPerInteger); //Ensure that the current mpz_t integer actually fits inside of the mpn_array.
                    err = cudaMemcpy(
                        deviceDataArrayPtr + (i * limbsPerInteger),
                        mpzToCopy->_mp_d,
                        ABSIZ(mpzToCopy) * sizeof(mp_limb_t),
                        cudaMemcpyHostToDevice
                    );

                    if (err != cudaSuccess) {
                        return err;
                    }

                    err = cudaMemcpy(
                        deviceSizesArrayPtr + i,
                        &SIZ(mpzToCopy),
                        sizeof(int),
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

    //Can be called from either the host or device.
    //Initializes a single integer inside of an mpn_array struct by copying the value of a limb into it.
    //DOES NOT PRESERVE NEGATIVITY.
    ANYCALLER void mpn_array_init_idx_set_limb(mpn_array* array, const int idx, const mp_limb_t value) {
        MPN_ARRAY_DATA(array)[idx * array->numLimbsPerInteger] = value;
        MPN_ARRAY_SIZES(array)[idx] = (value != 0);
    }
    //Can be called from either the host or device.
    //Initializes a single integer inside of an mpn_array struct by copying the value of a signed long long into it.
    //Preserves negativity.
    ANYCALLER void mpn_array_init_idx_set_sll(mpn_array* array, const int idx, const long long value) {
        MPN_ARRAY_DATA(array)[idx * array->numLimbsPerInteger] = value;
        MPN_ARRAY_SIZES(array)[idx] = SGN(value);
    }
    //Can be called from either the host or device.
    //Initializes a single integer inside of an mpn_array struct by copying the value of a signed integer into it.
    //Preserves negativity.
    ANYCALLER void mpn_array_init_idx_set_si(mpn_array* array, const int idx, const int value) {
        MPN_ARRAY_DATA(array)[idx * array->numLimbsPerInteger] = value;
        MPN_ARRAY_SIZES(array)[idx] = SGN(value);
    }


    namespace device {

        //Initializes a single integer inside of an mpn_array struct to zero.
        GPUONLY void mpn_array_init_idx(const mpn_device_array array, const int idx) {
            //Go through each limb that this integer occupies and set its value to 0.
            mp_limb_t* dataArray = MPN_ARRAY_DATA(array);
            for (int i = idx * array->numLimbsPerInteger; i < (idx + 1) * array->numLimbsPerInteger; i++) {
                dataArray[i] = 0;
            }
            //Set the size of this integer to 0 limbs since its value is now 0.
            MPN_ARRAY_SIZES(array)[idx] = 0;
        }

        //Initializes a new mpn_array struct on the current CUDA device, this is used after a struct has been allocated in order to zero its memory out.
        //This is VERY SLOW due to the fact that it has to zero out every limb in the array, and if I remember correctly memset on CUDA kernels forces serial execution.
        //It is heavily recommended to do this kind of initialization on the host side instead of inside a kernel.
        GPUONLY void mpn_array_init_SLOW(const mpn_device_array array) {
            //Go through each limb in the array and set it to 0.
            memset(MPN_ARRAY_DATA(array), 0, array->numIntegersInArray * array->numLimbsPerInteger * sizeof(mp_limb_t));
            //Go through each size in the array and set it to 0, since every number in the array is now 0 limbs long.
            memset(MPN_ARRAY_SIZES(array), 0, array->numIntegersInArray * sizeof(int));
        }

    }

}