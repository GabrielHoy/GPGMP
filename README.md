# GPGMP: The GPU-Accelerated GNU Multiple Precision Arithmetic Library

# ! IN PROGRESS: ANYTHING AND EVERYTHING IS SUBJECT TO CHANGE AND CODE WILL BE VERY MESSY !

## This is a library built ontop of the GNU MP library, intended to enable mass-parallelized GPU computation of arbitrary precision numbers.

I will write a better README(and docs) if and when I flesh out this library to a point where I feel comfortable publishing it as "production usable" code...

Current functionality:
- GPU-Compatible `mpn` routines, under `gpgmp::mpnRoutines::gp<normal_mpn_routine_name>` *(As of 06/10/2025 These have not been optimized for Warp Divergence, Memory Coalescence or Dynamic Allocations. I have only ported them to CUDA C; there are many optimizations to be had yet.)*
- `gpgmp::mpn_device_array` and `gpgmp::mpn_host_array` types which operate similarly to mpz_t[] array's, with the benefit of optimized Memory Coalescence and GPU compatibility.

TODO:
- GPU-Compatible `mpf` routines
- `gpgmp::mpf_device_array` and `gpgmp::mpf_host_array` types for GPU-compatible Floating Point numbers
- Optimize `gpmpn` routines, at least to get rid of Dynamic Allocations

```
The GPGMP Library is free software; you can redistribute it and/or modify
it under any terms that also adhere to the GNU MP library's terms - as of the time of writing these are:

* the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 3 of the License, or (at your
  option) any later version.

or

* the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any
  later version.

or both in parallel, as here.

The GPGMP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.
```