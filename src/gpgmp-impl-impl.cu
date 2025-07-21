//gpgmp-impl-impl
//lol
#include "gpgmp-impl.cuh"

__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void *__gpgmp_tmp_reentrant_alloc(struct tmp_reentrant_t **markp, size_t size)
{
  char    *p;
  size_t  total_size;

#define P   ((struct tmp_reentrant_t *) p)

  total_size = size + HSIZ;
  p = __GMP_ALLOCATE_FUNC_TYPE (total_size, char);
  P->size = total_size;
  P->next = *markp;
  *markp = P;
  return p + HSIZ;
} ATTRIBUTE_MALLOC;


__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_tmp_reentrant_free(struct tmp_reentrant_t *mark)
{
  struct tmp_reentrant_t  *next;

  while (mark != NULL)
    {
      next = mark->next;
      (*__gpgmp_free_func) ((char *) mark, mark->size);
      mark = next;
    }
}

__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void gpgmp_init_primesieve(gmp_primesieve_t *ps)
{
  ps->s0 = 0;
  ps->sqrt_s0 = 0;
  ps->d = SIEVESIZE;
  ps->s[SIEVESIZE] = 0;		/* sentinel */
}



__GPGMP_DECLSPEC __GPGMP_CALLERTYPE unsigned long int gpgmp_nextprime(gmp_primesieve_t *ps)
{
  unsigned long p, d, pi;
  unsigned char *sp;
  static unsigned char addtab[] =
    { 2,4,2,4,6,2,6,4,2,4,6,6,2,6,4,2,6,4,6,8,4,2,4,2,4,8,6,4,6,2,4,6,2,6,6,4,
      2,4,6,2,6,4,2,4,2,10,2,10 };
  unsigned char *addp = addtab;
  unsigned long ai;

  /* Look for already sieved primes.  A sentinel at the end of the sieving
     area allows us to use a very simple loop here.  */
  d = ps->d;
  sp = ps->s + d;
  while (*sp != 0)
    sp++;
  if (sp != ps->s + SIEVESIZE)
    {
      d = sp - ps->s;
      ps->d = d + 1;
      return ps->s0 + 2 * d;
    }

  /* Handle the number 2 separately.  */
  if (ps->s0 < 3)
    {
      ps->s0 = 3 - 2 * SIEVESIZE; /* Tricky */
      return 2;
    }

  /* Exhausted computed primes.  Resieve, then call ourselves recursively.  */

#if 0
  for (sp = ps->s; sp < ps->s + SIEVESIZE; sp++)
    *sp = 0;
#else
  memset (ps->s, 0, SIEVESIZE);
#endif

  ps->s0 += 2 * SIEVESIZE;

  /* Update sqrt_s0 as needed.  */
  while ((ps->sqrt_s0 + 1) * (ps->sqrt_s0 + 1) <= ps->s0 + 2 * SIEVESIZE - 1)
    ps->sqrt_s0++;

  pi = ((ps->s0 + 3) / 2) % 3;
  if (pi > 0)
    pi = 3 - pi;
  if (ps->s0 + 2 * pi <= 3)
    pi += 3;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 3;
    }

  pi = ((ps->s0 + 5) / 2) % 5;
  if (pi > 0)
    pi = 5 - pi;
  if (ps->s0 + 2 * pi <= 5)
    pi += 5;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 5;
    }

  pi = ((ps->s0 + 7) / 2) % 7;
  if (pi > 0)
    pi = 7 - pi;
  if (ps->s0 + 2 * pi <= 7)
    pi += 7;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 7;
    }

  p = 11;
  ai = 0;
  while (p <= ps->sqrt_s0)
    {
      pi = ((ps->s0 + p) / 2) % p;
      if (pi > 0)
	pi = p - pi;
      if (ps->s0 + 2 * pi <= p)
	  pi += p;
      sp = ps->s + pi;
      while (sp < ps->s + SIEVESIZE)
	{
	  *sp = 1, sp += p;
	}
      p += addp[ai];
      ai = (ai + 1) % 48;
    }
  ps->d = 0;
  return gpgmp_nextprime (ps);
}


__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_assert_header(const char *filename, int linenum)
{
  if (filename != NULL && filename[0] != '\0')
    {
      printf("%s:", filename);
      if (linenum != -1)
        printf("%d: ", linenum);
    }
};


__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_assert_fail (const char *filename, int linenum, const char *expr)
{
  __gpgmp_assert_header (filename, linenum);
  printf("GNU MP [GPGMP] assertion failed: %s\n", expr);
  //abort();
};



__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_exception(int error_bit)
{
    gpgmp_errno |= error_bit;
    #ifdef SIGFPE
    raise (SIGFPE);
    #else
    //__gpgmp_junk = 10 / __gpgmp_0;
    printf("GPGMP: EXCEPTION RAISED: NORMALLY WOULD DIVIDE BY 0 INTENTIONALLY HERE. . .\n");
    #endif
    //abort ();
};
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_divide_by_zero(void) {
    __gpgmp_exception (GMP_ERROR_DIVISION_BY_ZERO);
};
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_sqrt_of_negative(void) {
    __gpgmp_exception (GMP_ERROR_SQRT_OF_NEGATIVE);
};
__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_overflow_in_mpz(void) {
    __gpgmp_exception (GMP_ERROR_MPZ_OVERFLOW);
};

__GPGMP_DECLSPEC __GPGMP_CALLERTYPE void __gpgmp_invalid_operation(void) {
    //raise (SIGFPE);
    //abort ();
    printf("GPGMP: INVALID OPERATION - NORMALLY WOULD RAISE A SIGFPE AND ABORT HERE. . .\n");
};


#ifdef __CUDA_ARCH__
__GPGMP_DECLSPEC __device__ int __gpgmp_junk;
__GPGMP_DECLSPEC __device__ extern int gpgmp_errno;
#else
__GPGMP_DECLSPEC extern int gpgmp_errno;
__GPGMP_DECLSPEC int __gpgmp_junk;
#endif


#ifdef __CUDA_ARCH__
__device__
#endif
__GPGMP_DECLSPEC char __gpgmp_rands_initialized = 0;
#ifdef __CUDA_ARCH__
__device__
#endif
__GPGMP_DECLSPEC gmp_randstate_t  __gpgmp_rands;


__GPGMP_DECLSPEC void *__gpgmp_free_func(void *blk_ptr, size_t blk_size)
{
#ifdef DEBUG
    {
        mp_ptr p = blk_ptr;
        if (blk_size != 0)
        {
            if (p[-1] != (0xdeadbeef << 31) + 0xdeafdeed)
            {
                printf("gmp [GPGMP]: (free) data clobbered before allocation block\n");
                // abort();
            }
            if (blk_size % GMP_LIMB_BYTES == 0)
                if (p[blk_size / GMP_LIMB_BYTES] != ~((0xdeadbeef << 31) + 0xdeafdeed))
                {
                    printf("gmp [GPGMP]: (free) data clobbered after allocation block\n");
                    // abort();
                }
        }
        blk_ptr = p - 1;
    }
#endif
    free(blk_ptr);
    return nullptr;
}


__GPGMP_DECLSPEC void *__gpgmp_allocate_func(size_t size)
{
    void *ret;
#ifdef DEBUG
    size_t req_size = size;
    size += 2 * GMP_LIMB_BYTES;
#endif
    ret = malloc(size);
    if (ret == 0)
    {
        printf("GNU MP [GPGMP]: Cannot allocate memory (size=%lu)\n", (long)size);
        // abort();
    }

#ifdef DEBUG
    {
        mp_ptr p = ret;
        p++;
        p[-1] = (0xdeadbeef << 31) + 0xdeafdeed;
        if (req_size % GMP_LIMB_BYTES == 0)
            p[req_size / GMP_LIMB_BYTES] = ~((0xdeadbeef << 31) + 0xdeafdeed);
        ret = p;
    }
#endif
    return ret;
}

__GPGMP_DECLSPEC void *__gpgmp_reallocate_func(void *oldptr, size_t old_size, size_t new_size)
{
    void *ret;

#ifdef DEBUG
    size_t req_size = new_size;

    if (old_size != 0)
    {
        mp_ptr p = oldptr;
        if (p[-1] != (0xdeadbeef << 31) + 0xdeafdeed)
        {
            printf("gmp [GPGMP]: (realloc) data clobbered before allocation block\n");
            // abort();
        }
        if (old_size % GMP_LIMB_BYTES == 0)
            if (p[old_size / GMP_LIMB_BYTES] != ~((0xdeadbeef << 31) + 0xdeafdeed))
            {
                printf("gmp [GPGMP]: (realloc) data clobbered after allocation block\n");
                // abort();
            }
        oldptr = p - 1;
    }

    new_size += 2 * GMP_LIMB_BYTES;
#endif

    ret = realloc(oldptr, new_size);
    if (ret == 0)
    {
        printf("GNU MP [GPGMP]: Cannot reallocate memory (old_size=%lu new_size=%lu)\n", (long)old_size, (long)new_size);
        // abort();
    }

#ifdef DEBUG
    {
        mp_ptr p = ret;
        p++;
        p[-1] = (0xdeadbeef << 31) + 0xdeafdeed;
        if (req_size % GMP_LIMB_BYTES == 0)
            p[req_size / GMP_LIMB_BYTES] = ~((0xdeadbeef << 31) + 0xdeafdeed);
        ret = p;
    }
#endif
    return ret;
}