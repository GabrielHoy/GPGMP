# **GPGMP:** The GPU-Accelerated GNU Multiple Precision Arithmetic Library

# IN PROGRESS: ANYTHING AND EVERYTHING IS SUBJECT TO CHANGE AND CODE WILL BE VERY MESSY

## This is a library built ontop of the GNU MP library, intended to enable mass-parallelized GPU computation of arbitrary precision numbers.

I will write a more complete README - and docs - if/when I flesh out this library to a point where I feel comfortable publishing it as "production usable" code...

> **CURRENT FUNCTIONALITY:**
- GPU-Compatible `mpn` routines, under `gpgmp::mpnRoutines::gp<normal_mpn_routine_name>` *(As of 07/04/2025 These have not been optimized for Warp Divergence or Memory Coalescence. I have primarily focused on porting to CUDA C; there are many optimizations to be had yet.)*
- GPU-Compatible `mpf` routines, under `gpgmp::mpfRoutines::gp<normal_mpf_routine_name>` *(As of 07/04/2025 These have not been optimized for Warp Divergence or Memory Coalescence. I have primarily focused on porting to CUDA C; there are many optimizations to be had yet.)*
- Routines tailored to working with `mpf_array`'s and their optimized memory format for the GPU, located under `gpgmp::mpfArrayRoutines::gp<equivalent_mpf_routine_name>` *(As of 07/12/2025 These have not been optimized for Warp Divergence or Memory Coalescence. I have primarily focused on porting to CUDA C; there are many optimizations to be had yet.)*
- `gpgmp::mpn_device_array` and `gpgmp::mpn_host_array` types which operate similarly to mpz_t[] array's, with the benefit of optimized Memory Coalescence and GPU compatibility.
- `gpgmp::mpf_device_array` and `gpgmp::mpf_host_array` types which operate similarly to mpf_t[] array's, with the benefit of optimized Memory Coalescence and GPU compatibility.

> **KNOWN ISSUES:**
- There is scaffolding currently setup for "availableOperations" on `mpf_array`'s, but no actual assertations nor checks are performed to make sure that the user actually declared their 'intent' to use a function - and thereby whether necessary scratch space was pre-allocated - during `mpf_array` routines.
- `gpmpf_pow_ui` requires special logic to accomodate ahead-of-time scratch space allocation depending on the maximum exponent the user desires to raise something to the power of.
- Using `mpf` routines can be clunky and unintuitive at the moment due to some routines requiring dedicated scratch space to be allocated by the user ahead of time. There is no doubt a way to abstract this allocation.
- A few `mpn` routines don't currently play well with the GPU and can cause kernels to hang. Need to properly test mpn functions in the future to find all occurances of this and refactor.
- Unit Tests have not been written yet.

> **TODO:**
- Refactor several `mpn` routines to use pre-allocated scratch space instead of trying to dynamically allocate on the GPU
- Optimize `gpgmp::mpnRoutines` routines for parallelized processing
- Optimize `gpgmp::mpfRoutines` routines for parallelized processing
- Optimize `gpgmp::mpfArrayRoutines` routines for parallelized processing
- Write Usage Documentation
- Ensure no possible legal issues exist with this library extending off of GMP. *(Until I get to this, if you're with the GMP legal team and have any concerns feel free to reach out at legal@tamperedreality.net!)*
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

> FOR CONTRIBUTORS

- Some background on me which may explain oddities in the way I code; believe it or not I am an amateur C++ programmer, even newer than that at CUDA, and self taught! I work in a professional capacity with higher level languages like Luau for game dev, so for me this project is certainly diving into the deep end of low-level programming. You may see the ramnifications of that in the library's code, it may be very non-standard or messy in comparision to modern C++ programming practices or paradigms.
- If you're browsing the code and see me committing any 'mortal sins' - or have a suggestion about an optimization - by all means make a fuss with an issue/PR, I'm always trying to learn more and welcome contributions!