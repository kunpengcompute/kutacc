#pragma once
#include <iostream>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cstring>
#define SVL_fp32 16
#define SVL_bf16 32
#define SME_ON()                                                                                \
    __asm__ volatile("SMSTART":::"z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                \
                                 "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",          \
                                 "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",        \
                                 "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",        \
                                 "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",                \
                                 "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");         \
    __asm__ volatile("ISB")
#define SME_OFF()                                                                               \
    __asm__ volatile("SMSTOP":::"z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                 \
                                "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",           \
                                "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",         \
                                "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",         \
                                "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",                 \
                                "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");          \
    __asm__ volatile("ISB")


inline void prefetch_L2(const void *data) {
    __asm__ __volatile__(
        "prfm PLDL2KEEP, [%[data]]         \n\t"
        :: [data] "r" (data));
}

inline void prefetch_L1(const void *data) {
    __asm__ __volatile__(
        "prfm PLDL1STRM, [%[data]]         \n\t"
        :: [data] "r" (data));
}