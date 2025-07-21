# **GPGMP:** The GPU-Accelerated GNU Multiple Precision Arithmetic Library

## This is a library built ontop of the GNU MP library, intended to enable mass-parallelized GPU computation of arbitrary precision numbers.

**IN PROGRESS:** ANYTHING AND EVERYTHING IS SUBJECT TO CHANGE - CODE MAY NOT BE THUROUGHLY TESTED; I have currently only confirmed the following functions to work as expected across the CPU and GPU:
- Addition
- Subtraction
- Multiplication
- Division
- Square Roots

This library is **not** something I would consider "production usable" code quite yet, but I am publishing it as-is for anyone who would like to use it regardless or help contribute to the project.

> **CURRENT FUNCTIONALITY:**
- GPU-Compatible `mpn` routines, under `gpgmp::mpnRoutines::gp<normal_mpn_routine_name>`
- GPU-Compatible `mpf` routines, under `gpgmp::mpfRoutines::gp<normal_mpf_routine_name>`
- Routines tailored to working with `mpf_array`'s and their optimized memory format for the GPU, located under `gpgmp::mpfArrayRoutines::gp<equivalent_mpf_routine_name>`
- `gpgmp::mpn_device_array` and `gpgmp::mpn_host_array` types which operate similarly to mpz_t[] array's, with the benefit of optimized Memory Coalescence and GPU compatibility.
- `gpgmp::mpf_device_array` and `gpgmp::mpf_host_array` types which operate similarly to mpf_t[] array's, with the benefit of optimized Memory Coalescence and GPU compatibility.

> **KNOWN ISSUES:**
- Using GPGMP is not well documented for end-users yet, if you are not well acquainted with GMP and its usage I *heavily* recommend reading up on the base library before using GPGMP; it will save you lots of headache.
- None of the routines have been *particularly* optimized to avoid Warp Divergence yet. I have primarily focused on porting this massive library to CUDA C and resolving the myriad of implications that running GMP on the GPU introduces; there are many optimizations to be had yet and this library is certainly not as fast as it *could* be.
- availableOperations needs metadata associated with it instead of being a simple bit-field so we can reduce overall scratch space needed for some operations -- for example, mpf_div requires approx. ~12x the limb-count of scratch space in most cases currently, but this number *can* go much lower if the user doesn't plan to divide a numerator by a single-limb number.
- `mpfArrayRoutines::gpmpf_pow_ui` requires special logic to accomodate ahead-of-time scratch space allocation depending on the maximum exponent the user desires to raise something to the power of.
- Using `mpf` routines can be clunky and unintuitive at the moment due to some routines requiring dedicated scratch space to be allocated by the user ahead of time. There is no doubt a way to abstract this allocation.
- A few `mpn` routines don't currently play well with the GPU and can cause kernels to hang. Need to properly test mpn functions in the future to find all occurances of this and refactor.
- Multiplication routines currently use basecase multiplication instead in all cases instead of attempting to branch off into Toom or FFT multiplication methods - this may significantly slow down or speed up GPU multiplication due to warp divergence, need to investigate and profile both cases
- Unit Tests have not been written yet.

> **TODO:**
- Actually set this up with build steps etc to be a library instead of compiling to an executable for ease of testing
- Write some basic documentation for end-users to introduce them to the library and its basic usage
- Refactor __**many**__ `mpn` routines to use pre-allocated scratch space instead of trying to dynamically allocate on the GPU:
  - divexact.cu
  - divrem.cu
  - fib2_ui.cu
  - fib2m.cu
  - gcd.cu
  - gcdext.cu
  - get_str.cu
  - hgcd_reduce.cu
  - jacobi.cu
  - mu_div_q.cu
  - mul_fft.cu
  - mullo_n.cu
  - mulmid_n.cu
  - mulmid.cu
  - mulmod_bnm1.cu
  - nussbaumer_mul.cu
  - perfpow.cu
  - perfsqr.cu
  - powlo.cu
  - powm.cu
  - redc_n.cu
  - remove.cu
  - rootrem.cu
  - set_str.cu
  - sqrmod_bnm1.cu
  - strongfibo.cu
  - toom6_sqr.cu
  - toom6h_mul.cu
  - toom8_sqr.cu
  - toom8h_mul.cu
  - toom42_mul.cu
  - toom53_mul.cu
  - toom62_mul.cu
  - toom63_mul.cu
- Investigate whether `_alloca` is viable to run on the GPU; possibly refactor more `mpn` functions if not
- Optimize `gpgmp::mpnRoutines` routines for parallelized processing
- Optimize `gpgmp::mpfRoutines` routines for parallelized processing
- Optimize `gpgmp::mpfArrayRoutines` routines for parallelized processing
- Ensure no possible legal issues exist with this library extending off of GMP. *(Until I get to this, if you're with the GMP legal team and have any concerns feel free to reach out at legal@tamperedreality.net!)*
- Test all randomization functions to ensure they work on the GPU
- Generally clean the codebase up, standardize used naming conventions, remove zombie code left over from porting, etc.


> **FAQ:**
- What is GMP?
  - GMP is the GNU Multiple Precision Arithmetic Library; it allows for developers to perform arithmetic operations on arbitrarily large or small numbers which is normally not possible with 16/32/64 bit number types usually exposed to developers.
- Where's MPQ?
  - I've chosen not to create support for rationals in this library for now, in order to save time. If you're interested in a future implementation for rationals please let me know!
- Why did you choose MPF instead of trying to port MPFR?
  - MPFR is a big library on its own, and for the sake of simplicity I've chosen to limit my scope for an initial release of this project to purely the GMP library. In the future if I have time and there is interest, I may approach adding an MPFR port to this project in the future as well.
- This library is out of date with the current version of GMP!
  - At the time of this libraries creation the current GMP latest version is 6.3.0; this is the version that was used during development of GPGMP and I am unsure as to past/future GMP version interoperability with GPGMP. *GPGMP will likely not be kept up-to-date with future GMP versions as they release.*
- Why run GMP on the GPU in the first place?
  - I decided to create this project primarily to fill a technical void I noticed where noone else had created an implementation of Arbitrary-Precision Floating Point arithmetic on the GPU yet; the CUMP library does exist for basic floating point operations but I noticed it did not play well with Windows and didn't have some desired functionalities from GMP I needed for my other personal projects.

> FOR CONTRIBUTORS

- If you decide you'd like to contribute to the project, I'd more than welcome some help! Some of the most important items to tackle as it stands right now are:
  - Rewriting many of the `mpn` routines to take pre-allocated scratch space instead of attempting dynamic allocation mid-routines
  - Writing Unit Tests to ensure all `mpn`, `mpf` and `mpfArray` routines can run inside of GPU Kernels successfully
  - Abstracting away the necessity for users to pre-allocate scratch space when they want to use `mpn` routines -- *possibly implement our own idea of an `mpz_array` which pre-allocates space, similar to the `mpf_array`'s?*
- This project is one of my first deep-dives into CUDA C and its intricacies; You may see the ramnifications of that in the library's code, if it is non-standard or messy in comparision to "modern" C++/CUDA C paradigms feel free to make an issue and I'll try to correct it!
- If you're browsing the code and see me committing any 'mortal sins' - or have a suggestion about an optimization - by all means make a fuss with an issue/PR, I'm always trying to learn more and welcome contributions!