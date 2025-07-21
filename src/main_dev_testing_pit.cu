#include <iostream>
#include <string>
#include "gpgmp.cuh"

#define ARRAY_LENGTH 5
#define PRECISION_BITS 64*5

ANYCALLER void PrintDataAboutMPFArray_AsDoubles(gpgmp::mpf_array* testArray) {
    for (int i = 0; i < testArray->numFloatsInArray; i++) {
        printf("Array[%d] currently has a double value of %f!\n", i, gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, i}));
    }
}

void PrintDataAboutMPFArray_WithDigits(gpgmp::mpf_array* testArray, int digitsToPrint = 10) {
    mpf_t* gmpArray = new mpf_t[testArray->numFloatsInArray];
    gpgmp::host::mpf_array_copy_to_gmp_mpf_array_with_mpf_init(gmpArray, testArray);
    for (int i = 0; i < testArray->numFloatsInArray; i++) {
        gmp_printf((std::string("Array[%d] currently has a double value of %.") + std::to_string(digitsToPrint) + std::string("Ff!\n")).c_str(), i, gmpArray[i]);//gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, i}));
    }

    for (int i = 0; i < testArray->numFloatsInArray; i++) {
        mpf_clear(gmpArray[i]);
    }
    delete[] gmpArray;
}

__global__ void initKernel() {
    printf("CUDA Initialized...\n");
}

#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\nError detected during check on line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)








__global__ void testKernel() {
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (ARRAY_LENGTH - 1)) {
        return;
    }
    printf("[GPU]: Hello, world! I'm a kernel with a thread identifier of %d!\n", threadIdentifier);
}

__global__ void printGeneralArrayData(gpgmp::mpf_device_array testArray)
{
    printf("[GPU]: Data about array:\n");
    printf("[GPU]:     - availableOperations: %u\n", testArray->availableOperations);
    printf("[GPU]:     - userSpecifiedPrecisionLimbCount: %u\n", testArray->userSpecifiedPrecisionLimbCount);
    printf("[GPU]:     - limbsPerArrayFloat: %u\n", testArray->limbsPerArrayFloat);
    printf("[GPU]:     - numFloatsInArray: %u\n", testArray->numFloatsInArray);
}

__global__ void setArrayData(gpgmp::mpf_device_array testArray)
{
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (ARRAY_LENGTH - 1)) {
        return;
    }

    double toSetTo = static_cast<double>(threadIdentifier)+1.1337;
    gpgmp::mpfArrayRoutines::gpmpf_set_d({testArray, threadIdentifier}, toSetTo);
    printf("[GPU]: Array[%d] set to %f!\n", threadIdentifier, toSetTo);
}

__global__ void printCurrentMpfArrayData(gpgmp::mpf_device_array testArray)
{
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (ARRAY_LENGTH - 1)) {
        return;
    }

    printf("[GPU]: Array[%d] currently has a double value of %f!\n", threadIdentifier, gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier}));
}

__global__ void setSpecificArrayIdxToValue(gpgmp::mpf_device_array testArray, int idx, double value)
{
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier != idx) {
        return;
    }

    gpgmp::mpfArrayRoutines::gpmpf_set_d({testArray, threadIdentifier}, value);
    printf("[GPU]: Array[%d] specifically set to have value %f!\n", threadIdentifier, value);
}

__global__ void performAddition(gpgmp::mpf_device_array testArray)
{
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (ARRAY_LENGTH - 1)) {
        return;
    }

    double valueBeforeOp = gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier});
    gpgmp::mpfArrayRoutines::gpmpf_add_ui({testArray, threadIdentifier}, {testArray, threadIdentifier}, 1);
    printf("[GPU]: Array[%d] Addition Result: %f -> %f!\n", threadIdentifier, valueBeforeOp, gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier}));
}

__global__ void performSubtraction(gpgmp::mpf_device_array testArray)
{
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (ARRAY_LENGTH - 1)) {
        return;
    }

    double valueBeforeOp = gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier});
    gpgmp::mpfArrayRoutines::gpmpf_sub_ui({testArray, threadIdentifier}, {testArray, threadIdentifier}, 1);
    printf("[GPU]: Array[%d] Subtraction Result: %f -> %f!\n", threadIdentifier, valueBeforeOp, gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier}));
}

__global__ void performMultiplication(gpgmp::mpf_device_array testArray)
{
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (ARRAY_LENGTH - 1)) {
        return;
    }

    double valueBeforeOp = gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier});
    gpgmp::mpfArrayRoutines::gpmpf_mul_ui({testArray, threadIdentifier}, {testArray, threadIdentifier}, 2);
    printf("[GPU]: Array[%d] Multiplication Result: %f -> %f!\n", threadIdentifier, valueBeforeOp, gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier}));
}

__global__ void performDivision(gpgmp::mpf_device_array testArray)
{
    int threadIdentifier = (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIdentifier > (ARRAY_LENGTH - 1)) {
        return;
    }

    double valueBeforeOp = gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier});
    //gpgmp::mpfArrayRoutines::gpmpf_div_ui({testArray, threadIdentifier}, {testArray, threadIdentifier}, 2);
    gpgmp::mpfArrayRoutines::gpmpf_div({testArray, threadIdentifier}, {testArray, threadIdentifier}, {testArray, ARRAY_LENGTH-1});
    //gpgmp::mpfArrayRoutines::gpmpf_ui_div({testArray, threadIdentifier}, 2, {testArray, threadIdentifier});
    printf("[GPU]: Array[%d] Division Result: %f -> %f!\n", threadIdentifier, valueBeforeOp, gpgmp::mpfArrayRoutines::gpmpf_get_d({testArray, threadIdentifier}));
}


int main(int argc, char** argv) {
    printf("Initializing CUDA on the GPU...Kernel will run momentarily...\n");
    initKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    printf("General program data:\n   - %u numbers\n   - %u bits(%u limbs) of precision...\nExecuting Kernel...\n\n", ARRAY_LENGTH, PRECISION_BITS, LIMB_COUNT_FROM_PRECISION_BITS(PRECISION_BITS));


    { //CPU Side Tests
        printf("Setting up GPGMP data for test...\n");
        gpgmp::mpf_host_array testArray;
        bool success = gpgmp::host::mpf_array_allocate_on_host(
            testArray,
            ARRAY_LENGTH,
            PRECISION_BITS,
            gpgmp::OP_ALL
        );
        if (!success) {
            printf("Failed to allocate array on host!\n");
            return 1;
        }

        printf("Array allocated!\n");

        gpgmp::host::mpf_array_init(testArray);
        printf("Array initialized!\n");

        printf("Data about array:\n");
        printf("    - availableOperations: %u\n", testArray->availableOperations);
        printf("    - userSpecifiedPrecisionLimbCount: %u\n", testArray->userSpecifiedPrecisionLimbCount);
        printf("    - limbsPerArrayFloat: %u\n", testArray->limbsPerArrayFloat);
        printf("    - numFloatsInArray: %u\n", testArray->numFloatsInArray);

        for (int i = 0; i < testArray->numFloatsInArray; i++) {
            double toSetTo = static_cast<double>(i)+1.1337;
            gpgmp::mpfArrayRoutines::gpmpf_set_d({testArray, i}, toSetTo);
            printf("Array[%d] set to %f!\n", i, toSetTo);
        }


        PrintDataAboutMPFArray_WithDigits(testArray, 6);

        printf("\nTesting Addition...Will add 1 to each array element...\n\n");
        for (int i = 0; i < testArray->numFloatsInArray; i++) {
            gpgmp::mpfArrayRoutines::gpmpf_add_ui({testArray, i}, {testArray, i}, 1);
        }

        PrintDataAboutMPFArray_WithDigits(testArray, 6);

        printf("\nTesting Subtraction...Will subtract 1 from each array element...\n\n");
        for (int i = 0; i < testArray->numFloatsInArray; i++) {
            //gpgmp::mpfArrayRoutines::gpmpf_sub({testArray, i}, {testArray, i}, {testArray, ARRAY_LENGTH-1});
            gpgmp::mpfArrayRoutines::gpmpf_sub_ui({testArray, i}, {testArray, i}, 1);
        }

        PrintDataAboutMPFArray_WithDigits(testArray, 6);

        printf("\nTesting Multiplication...Will multiply each array element by 2...\n\n");
        for (int i = 0; i < testArray->numFloatsInArray; i++) {
            gpgmp::mpfArrayRoutines::gpmpf_mul_ui({testArray, i}, {testArray, i}, 2);
            //gpgmp::mpfArrayRoutines::gpmpf_mul({testArray, i}, {testArray, i}, {testArray, ARRAY_LENGTH-1});
        }

        PrintDataAboutMPFArray_WithDigits(testArray, 6);

        printf("\nTesting Division...Will divide each array element by 2...\n\n");
        for (int i = 0; i < testArray->numFloatsInArray; i++) {
            //gpgmp::mpfArrayRoutines::gpmpf_div({testArray, i}, {testArray, i}, {testArray, ARRAY_LENGTH-1});
            gpgmp::mpfArrayRoutines::gpmpf_div_ui({testArray, i}, {testArray, i}, 2);
        }

        PrintDataAboutMPFArray_WithDigits(testArray, 6);
    }

    printf("\n\n\nGPU Time!\n\n\n\n");
    cudaError_t err;
    int gridDimXandY = static_cast<int>(ceil(sqrt(ceil(static_cast<float>(ARRAY_LENGTH) / 512.f))));
    dim3 launchDims = dim3(gridDimXandY, gridDimXandY, 1);
    dim3 blockDims = dim3(512, 1, 1);

    gpgmp::mpf_device_array testArrayGPU;
    err = gpgmp::host::mpf_array_allocate_on_device(testArrayGPU, ARRAY_LENGTH, PRECISION_BITS, gpgmp::OP_ALL);
    CHECK_CUDA_ERROR(err);


    printf("GPU Array allocated! Beginning Kernel Launches...\n");

    printGeneralArrayData<<<1,1>>>(testArrayGPU);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\n");

    setArrayData<<<launchDims, blockDims>>>(testArrayGPU);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printCurrentMpfArrayData<<<launchDims, blockDims>>>(testArrayGPU);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printf("Addition Kernel:\n");
    performAddition<<<launchDims, blockDims>>>(testArrayGPU);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printf("Subtraction Kernel:\n");
    performSubtraction<<<launchDims, blockDims>>>(testArrayGPU);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printf("Multiplication Kernel:\n");
    performMultiplication<<<launchDims, blockDims>>>(testArrayGPU);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printf("Setting last array element to 2 for division test:\n");
    setSpecificArrayIdxToValue<<<launchDims, blockDims>>>(testArrayGPU, ARRAY_LENGTH-1, 2.0);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printf("Division Kernel:\n");
    performDivision<<<launchDims, blockDims>>>(testArrayGPU);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("Current Last Error after Division Synchronization: %s\n", cudaGetErrorString(cudaGetLastError()));
    printf("\n");


    printf("Program run finished!\n");



    printf("\n\n\n\n\n\n");
    float maximumRatioIToN = 0.0f;
    mp_size_t occuredAtDenom = 0;
    mp_size_t occuredAtNum = 0;

    for (mp_size_t numeratorNumLimbs = 1; numeratorNumLimbs <= 10000; numeratorNumLimbs++)
    {
        for (mp_size_t denominatorNumLimbs = 1; denominatorNumLimbs <= numeratorNumLimbs; denominatorNumLimbs++)
        {
            mp_size_t itch = gpgmp::mpnRoutines::gpmpn_tdiv_qr_itch(numeratorNumLimbs, denominatorNumLimbs);
            float ratioIToN = (float)itch / (float)numeratorNumLimbs;
            if (ratioIToN > maximumRatioIToN)
            {
                maximumRatioIToN = ratioIToN;
                occuredAtDenom = denominatorNumLimbs;
                occuredAtNum = numeratorNumLimbs;
            }
        }
    }


    printf("Maximum ratio of I to N: %f, occured at denominatorNumLimbs: %d, numeratorNumLimbs: %d\n", maximumRatioIToN, occuredAtDenom, occuredAtNum);


    return 0;
}