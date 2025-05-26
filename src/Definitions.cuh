#pragma once

//Useful macros ported over from GMP
#define SGN(x)       ((x)<0 ? -1 : (x) != 0)
#define ABS(x)       ((x)>=0 ? (x) : -(x))
#undef MIN
#define MIN(l,o) ((l) < (o) ? (l) : (o))
#undef MAX
#define MAX(h,i) ((h) > (i) ? (h) : (i))
