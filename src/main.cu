#include <iostream>
#include "gpgmp.cuh"

#define NUM_INTEGERS_IN_ARRAY 8294400
#define PRECISION_PER_INTEGER 64*64
#define MULTIPLICATIONS_TO_PERFORM 100
#define NUM_RUNS_FOR_AVG_TIME_GATHERING 100

#define ABS(x) ((x) >= 0 ? (x) : -(x))
double ConvertBackToDouble(mpf_t val) {
    uint64_t intPart = 0;
    double fracPart = 0;

    //Important concepts:
    //  mp_d is the actual stored limb data inside of the mpf_t struct. It has a total size of mp_size limbs.
    //  The actual mp_d array can be conceptually split into two different arrays: an IntegerPart array and a FractionalPart array. FractionalPart comes first(if it exists! read on), IntegerPart comes second(if it exists).
    //  Both of these "sub-arrays" are in little-endian order, so the first limbs inside of them are their LEAST significant limbs.
    //  The mp_exp variable can be thought about as a number representing how many limbs to the LEFT the decimal point is. This means that FractionalPart's last index is at mp_size - mp_exp. MP_EXP CAN BE NEGATIVE! If it is, then the decimal point is even further to the right than the last limb by mp_exp limbs - this is used for values < 1 that are very, small in the fractional part - thus if mp_exp < 0 then there will never be an integer part.

    int integerArraySize = val->_mp_exp;
    int numLimbsUsed = ABS(val->_mp_size);
    int fractionalArraySize = numLimbsUsed - val->_mp_exp;

    //int fractionalArrayBeginningIdxInTotalArray = 0;
    int integerArrayBeginningIdxInTotalArray = numLimbsUsed - integerArraySize;

    mp_limb_t* fractionalSubArray = val->_mp_d; //simply mp_d since the fractional part comes first.
    mp_limb_t* integerSubArray = val->_mp_d + integerArrayBeginningIdxInTotalArray; //The integer array starts after the fractional array.


    //There's only an integer part to this value if integerArraySize(mp_exp) > 0 - otherwise, the entire array is a fractional part since the integer beginning being <=0 places to the left of the end of the array means that the entire array is a fractional part.
    if (integerArraySize > 0) {
        for (int intPartIdx = 0; intPartIdx < integerArraySize; intPartIdx++) {  // (val->_mp_size - (val->_mp_exp - 1)); intPartIdx < val->_mp_size; intPartIdx++) {
            intPart += integerSubArray[intPartIdx] * pow(2, GMP_NUMB_BITS * (intPartIdx)); //integerSubArray[intPartIdx] w/o simplified variable names expands to val->_mp_d[(val->_mp_size - val->_mp_exp) + intPartIdx]
        }
    }

    for (int fracPartIdx = 0; fracPartIdx < fractionalArraySize; fracPartIdx++) {
        mp_limb_t fracPartLimb = fractionalSubArray[fracPartIdx];
        double doubleFracPart = static_cast<double>(fracPartLimb);

        /*
            This variable is the "size contribution" of the current fractional limb;
            it is the amount of times which we need to shift the limb to the right by <bitsPerLimb>(i.e limb / 2^(GMP_NUMB_BITS*sizeContrib)) in order to transform the limb to...
            ...the correct decimal size to be able to add the limb's value to a double correctly.

            We decrement fracPartIdx from (mp_size - mp_exp) since:
              Ontop of this little bit of array concatenation confusion, we also need to keep in mind that each of these "sub-arrays" are in little-endian order, therefore to find the true "size contribution" of any limb we actually need...
              ...to invert the index relative to the end of its sub-array! So (endOfSubArray - currentIndexFromStartOfSubArray) will give us the true size contribution of any limb;
            Therefore (val->_mp_size(totalSizeOfBothArrays) - val->_mp_exp(sizeOfIntegerArray) is the size of FractionalPart; then we subtract fracPartIdx from that size to get the "size contribution"
        */
        int invertedFractionalLimbIndexStartingFromLastIndexInFractionalArray = fractionalArraySize - fracPartIdx;
        //We now have what we would 'normally' think of as the index of the current fractional limb in the fractional limb array, as if it were big endian.
        //This is useful because it represents how far the limb is from the right of the decimal point; specifically it means that we are <limbBits>*n bits to the right of the decimal point

        //...We can use that index as a multiplier for the division's power.
        int bitsToTheRightOfDecimalPointThatThisLimbRepresents = GMP_NUMB_BITS * invertedFractionalLimbIndexStartingFromLastIndexInFractionalArray;

        //This double is the actual value of the current fractional limb, converted to a double.
        double fracPartRepresentedByThisLimb = (doubleFracPart / (pow(2, bitsToTheRightOfDecimalPointThatThisLimbRepresents)));
        fracPart += fracPartRepresentedByThisLimb;
    }

    return (fracPart + static_cast<double>(intPart)) * SGN(val->_mp_size);
}

ANYCALLER void PrintDataAboutMPNArray(gpgmp::mpn_array* array) {
    printf("Array data(%d integers in array, %d limbs allocated per integer):\n", array->numIntegersInArray, array->numLimbsPerInteger);
    for (int i = 0; i < array->numIntegersInArray; i++) {
        printf("    Integer #%d", i);
        if (MPN_ARRAY_SIZES(array)[i] < 0) {
            printf("(NEGATIVE)");
        }
        printf(":\n");
        for (int limbIdx = 0; limbIdx < array->numLimbsPerInteger; limbIdx++) {
            printf("        Limb #%d: %llu\n", limbIdx, MPN_ARRAY_DATA(array)[i * array->numLimbsPerInteger + limbIdx]);
        }
    }
}

__global__ void testKernel(gpgmp::mpn_device_array deviceArray) {
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (NUM_INTEGERS_IN_ARRAY / 2) - 1) {
        return;
    }
    //printf("Hello, world! I'm a kernel with a thread index of %d!\n", threadIdentifier);

    //gpgmp::mpnRoutines::gpmpn_add_n(&(*deviceArray)[0], &(*deviceArray)[0], &(*deviceArray)[idxAdd], deviceArray->numLimbsPerInteger);
    //gpgmp::mpnRoutines::gpmpn_sub(&(*deviceArray)[0], &(*deviceArray)[1], 2, &(*deviceArray)[2], 2);
    //gpgmp::mpnRoutines::gpmpn_sqr(&(*deviceArray)[1], &(*deviceArray)[3], 2);
    int doubleThreadIdent = threadIdentifier * 2;
    gpgmp::mpn_array_init_idx_set_si(deviceArray, (doubleThreadIdent) + 1, 1000000);

    mp_limb_t multBy = 2;
    for (int i = 0; i < MULTIPLICATIONS_TO_PERFORM; i++) {
        gpgmp::mpnRoutines::gpmpn_mul(&(*deviceArray)[doubleThreadIdent], &(*deviceArray)[(doubleThreadIdent) + 1], (PRECISION_PER_INTEGER/64) - 1, &multBy, 1);
        //gpgmp::mpnRoutines::gpmpn_tdiv_qr(&(*deviceArray)[doubleThreadIdent], &(*deviceArray)[(doubleThreadIdent) + 1], 0, &(*deviceArray)[(doubleThreadIdent) + 1], 2, &multBy, 1);

        //gpgmp::mpnRoutines::gpmpn_copyd(&(*deviceArray)[(doubleThreadIdent) + 1], &(*deviceArray)[doubleThreadIdent], PRECISION_PER_INTEGER/64);
    }

    //gpgmp::mpnRoutines::gpmpn_mul(&(*deviceArray)[0], &(*deviceArray)[1], 2, &(*deviceArray)[2], 2);
    //gpgmp::mpnRoutines::gpmpn_lshift(&(*deviceArray)[1], &(*deviceArray)[1], 4, 2);
    //gpgmp::mpnRoutines::gpmpn_rshift(&(*deviceArray)[1], &(*deviceArray)[1], 4, 1);
    //gpgmp::mpnRoutines::gpmpn_tdiv_qr(&(*deviceArray)[3], &(*deviceArray)[0], 0, &(*deviceArray)[0], 2, &(*deviceArray)[1], 2);


    //PrintDataAboutMPNArray(deviceArray);
}

__global__ void printDeviceArrayData(gpgmp::mpn_device_array deviceArray) {
    PrintDataAboutMPNArray(deviceArray);
}

__global__ void initKernel() {
    printf("CUDA Initialized...\n");
}

#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\nError detected during check on line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)


int main(int argc, char** argv) {
    __gmpf_set_default_prec(64*5);
    cudaError_t err;

    gpgmp::mpn_device_array testArray;
    printf("Allocating array on GPU...\n");

    err = gpgmp::host::mpn_array_allocate_on_device(testArray, NUM_INTEGERS_IN_ARRAY, PRECISION_PER_INTEGER);
    CHECK_CUDA_ERROR(err);

    printf("Array allocated!\n");

    //printf("Initializing array on GPU from CPU...\n");
    //mpz_t mpzArray[NUM_INTEGERS_IN_ARRAY];
    //for (int i = 0; i < NUM_INTEGERS_IN_ARRAY; i++) {
    //    mpz_init_set_si(mpzArray[i], 1);
    //}
    //mpz_init_set_d(mpzArray[0], UINT64_MAX);
    //mpz_init_set_d(mpzArray[1], 1337.0);
    //mpz_init_set_d(mpzArray[2], 2.0);
    //mpz_init_set_d(mpzArray[0], 5.0);
    //mpz_init_set_d(mpzArray[1], 2.0);

    //err = gpgmp::host::mpn_array_init_on_device_from_mpz_array(testArray, mpzArray, NUM_INTEGERS_IN_ARRAY, PRECISION_PER_INTEGER, NUM_INTEGERS_IN_ARRAY);
    //CHECK_CUDA_ERROR(err);
    //printf("Array initialized!\n");

    printf("Initializing GPGMP on the GPU...Kernel will run momentarily...\n");
    initKernel<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("Executing Kernel - crunching %u integers with %u bits of precision per integer...\n", NUM_INTEGERS_IN_ARRAY * MULTIPLICATIONS_TO_PERFORM, PRECISION_PER_INTEGER);



    float avgTime = 0;

    for (int i = 0; i < NUM_RUNS_FOR_AVG_TIME_GATHERING; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        int gridDimXandY = static_cast<int>(ceil(sqrt(ceil(static_cast<float>(NUM_INTEGERS_IN_ARRAY) / 512.f))));
        testKernel<<<dim3(gridDimXandY, gridDimXandY, 1), dim3(512, 1, 1)>>>(testArray);
        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaDeviceSynchronize();
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
            exit(1);
        }

        printf("Kernel %u execution time: %.2f milliseconds\n", i, milliseconds);
        avgTime += milliseconds;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    avgTime /= static_cast<float>(NUM_RUNS_FOR_AVG_TIME_GATHERING);

    printf("Average execution time for kernels over %u runs: %.2f milliseconds\n", NUM_RUNS_FOR_AVG_TIME_GATHERING, avgTime);

    //printDeviceArrayData<<<1,1>>>(testArray);
    cudaDeviceSynchronize();
    printf("Kernel finished!\n");

    printf("Testing custom mpn_add_n...\n");
    mpz_t mpzResult, mpzTest1, mpzTest2;
    mpz_init(mpzResult);
    mpz_init(mpzTest1);
    mpz_init(mpzTest2);
    mpz_set_d(mpzResult, UINT64_MAX);
    mpz_set_d(mpzTest1, UINT64_MAX);
    mpz_set_d(mpzTest2, UINT64_MAX);

    mp_srcptr operand1_ptr = mpzTest1->_mp_d;
    mp_srcptr operand2_ptr = mpzTest2->_mp_d;
    mp_size_t size = mpzTest1->_mp_size;

    mp_limb_t carry = gpgmp::mpnRoutines::gpmpn_add_n(mpzResult->_mp_d, operand1_ptr, operand2_ptr, size);
    printf("(Final Carry = %llu)\n", carry);

    printf("(CPU) mpzTest1 = (%llu*(2^64)) + %llu\n", mpzTest1->_mp_d[1], mpzTest1->_mp_d[0]);
    printf("(CPU) mpzTest2 = (%llu*(2^64)) + %llu\n", mpzTest2->_mp_d[1], mpzTest2->_mp_d[0]);
    printf("(CPU) Result = (%llu*(2^64)) + %llu\n", mpzResult->_mp_d[1], mpzResult->_mp_d[0]);


    return 0;
}