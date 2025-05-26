#include <iostream>
#include <bit>
#include "gmp.h"
#include "gpgmp.cuh"

double ConvertBackToDouble(mpf_t val) {
    uint64_t intPart = 0;
    double fracPart = 0;

    //Important concepts:
    //  mp_d is the actual stored limb data inside of the mpf_t struct. It has a total size of mp_size limbs.
    //  The actual mp_d array can be conceptually split into two different arrays: an IntegerPart array and a FractionalPart array. FractionalPart comes first(if it exists! read on), IntegerPart comes second(if it exists).
    //  Both of these "sub-arrays" are in little-endian order, so the first limbs inside of them are their LEAST significant limbs.
    //  The mp_exp variable can be thought about as a number representing how many limbs to the LEFT the decimal point is. This means that FractionalPart's last index is at mp_size - mp_exp. MP_EXP CAN BE NEGATIVE! If it is, then the decimal point is even further to the right than the last limb by mp_exp limbs - this is used for values < 1 that are very, small in the fractional part - thus if mp_exp < 0 then there will never be an integer part.

    int integerArraySize = val->_mp_exp;
    int numLimbsUsed = abs(val->_mp_size);
    int fractionalArraySize = numLimbsUsed - val->_mp_exp;

    int fractionalArrayBeginningIdxInTotalArray = 0;
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

    return fracPart + static_cast<double>(intPart);
}

int main(int argc, char** argv) {
    __gmpf_set_default_prec(64*5);

    mpf_t test;
    mpf_init(test);
    mpf_t test2;
    mpf_init(test2);

    mpf_set_str(test, "0.1", 10);
    //mpf_set_str(test, "0.0100000000001", 10);


    //mpf_set_str(test2, "18446744073709551615", 10);
    //mpf_set_str(test2, "18446744073709551618", 10);
    //mpf_set_str(test2, "36893488147419103230", 10);
    //mpf_set_ui(test2, std::numeric_limits<uint64_t>::max()); //gets truncated or bounded or something at UINT32_MAX for some reason
    //mpf_set_ui(test2, 100000000000000); //10000000 is max before losing that last 1


    mpf_add(test, test, test2);

    gmp_printf("test: %F.14f\n", test);
    printf("MPF breakdown:\n");
    std::cout << "    Precision: " << test->_mp_prec << std::endl;
    std::cout << "    Size: " << test->_mp_size << std::endl;
    std::cout << "    Exponent: " << test->_mp_exp << std::endl;
    std::cout << "    Data:" << std::endl;
    mp_limb_t* limbs = test->_mp_d;

    for (mp_limb_t i = 0; i < test->_mp_size; i++) {
        std::cout << "       Limb " << i << ": " << limbs[i] << std::endl;
    }

    printf("\n\n\n\n");
    std::cout << "Bit size of mp_limb_t: " << sizeof(mp_limb_t) * 8 << std::endl;
    std::cout << "UINT32_MAX: " << UINT32_MAX << std::endl;
    std::cout << "UINT64_MAX: " << UINT64_MAX << std::endl;
    printf("\n\n\n\n");

    printf("Double value of test: ");
    double doubleVerOfTest = ConvertBackToDouble(test);
    printf("%.13f\n", doubleVerOfTest);

    return 0;
}