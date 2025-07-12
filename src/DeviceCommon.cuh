#pragma once

#include "gpgmp-impl.cuh"

namespace gpgmp {
    namespace internal {
        //Intended to be a device-compatible equivalent of strlen.
        ANYCALLER size_t cudaStrLen(const char *str)
        {
            #ifdef __CUDA_ARCH__
                size_t len = 0;
                while (str[len] != '\0')
                    ++len;
                return len;
            #else
                return strlen(str);
            #endif
        }

        //Intended to be a device-compatible equivalent of isspace.
        ANYCALLER bool cudaIsSpace(char c)
        {
            #ifdef __CUDA_ARCH__
                // Standard whitespace characters: ' ', '\t', '\n', '\v', '\f', '\r'
                return (c == ' ')  ||
                    (c == '\t') ||
                    (c == '\n') ||
                    (c == '\v') ||
                    (c == '\f') ||
                    (c == '\r');
            #else
                return isspace(static_cast<unsigned char>(c));
            #endif
        }
    }
}
