#include <iostream>
#include <string>
#include "gpgmp.cuh"

#define ARRAY_LENGTH 5
#define PRECISION_BITS 64*5

__global__ void initKernel() {
    printf("CUDA Initialized...\n");
}

#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\nError detected during check on line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)



int main(int argc, char** argv)
{
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
}