/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__SM_32_INTRINSICS_HPP__)
#define __SM_32_INTRINSICS_HPP__

#if defined(__CUDACC_RTC__)
#define __SM_32_INTRINSICS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_32_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 320

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

// In here are intrinsics which are built in to the compiler. These may be
// referenced by intrinsic implementations from this file.
extern "C"
{
    // There are no intrinsics built in to the compiler for SM-3.5,
    // all intrinsics are now implemented as inline PTX below.
}

/*******************************************************************************
*                                                                              *
*  Below are implementations of SM-3.5 intrinsics which are included as        *
*  source (instead of being built in to the compiler)                          *
*                                                                              *
*******************************************************************************/

// LDG is a "load from global via texture path" command which can exhibit higher
// bandwidth on GK110 than a regular LD.
// Define a different pointer storage size for 64 and 32 bit
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

/******************************************************************************
 *                                   __ldg                                    *
 ******************************************************************************/

// Size of long is architecture and OS specific.
#if defined(__LP64__) // 64 bits
__SM_32_INTRINSICS_DECL__ long __ldg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
#else // 32 bits
__SM_32_INTRINSICS_DECL__ long __ldg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
#endif


__SM_32_INTRINSICS_DECL__ char __ldg(const char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (char)ret; }
__SM_32_INTRINSICS_DECL__ signed char __ldg(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (signed char)ret; }
__SM_32_INTRINSICS_DECL__ short __ldg(const short *ptr) { unsigned short ret; asm volatile ("ld.global.nc.s16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return (short)ret; }
__SM_32_INTRINSICS_DECL__ int __ldg(const int *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (int)ret; }
__SM_32_INTRINSICS_DECL__ long long __ldg(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.nc.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long long)ret; }
__SM_32_INTRINSICS_DECL__ char2 __ldg(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.nc.v2.s8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ char4 __ldg(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.nc.v4.s8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ short2 __ldg(const short2 *ptr) { short2 ret; asm volatile ("ld.global.nc.v2.s16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ short4 __ldg(const short4 *ptr) { short4 ret; asm volatile ("ld.global.nc.v4.s16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int2 __ldg(const int2 *ptr) { int2 ret; asm volatile ("ld.global.nc.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int4 __ldg(const int4 *ptr) { int4 ret; asm volatile ("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ longlong2 __ldg(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.nc.v2.s64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ unsigned char __ldg(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.u8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr));  return (unsigned char)ret; }
__SM_32_INTRINSICS_DECL__ unsigned short __ldg(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.nc.u16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned int __ldg(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.nc.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned long long __ldg(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.nc.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uchar2 __ldg(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.nc.v2.u8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ uchar4 __ldg(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.nc.v4.u8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ ushort2 __ldg(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.nc.v2.u16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ushort4 __ldg(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.nc.v4.u16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint2 __ldg(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.nc.v2.u32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint4 __ldg(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ulonglong2 __ldg(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.nc.v2.u64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ float __ldg(const float *ptr) { float ret; asm volatile ("ld.global.nc.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double __ldg(const double *ptr) { double ret; asm volatile ("ld.global.nc.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float2 __ldg(const float2 *ptr) { float2 ret; asm volatile ("ld.global.nc.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float4 __ldg(const float4 *ptr) { float4 ret; asm volatile ("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double2 __ldg(const double2 *ptr) { double2 ret; asm volatile ("ld.global.nc.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }


/******************************************************************************
 *                                   __ldcg                                    *
 ******************************************************************************/

// Size of long is architecture and OS specific.
#if defined(__LP64__) // 64 bits
__SM_32_INTRINSICS_DECL__ long __ldcg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldcg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
#else // 32 bits
__SM_32_INTRINSICS_DECL__ long __ldcg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldcg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
#endif


__SM_32_INTRINSICS_DECL__ char __ldcg(const char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (char)ret; }
__SM_32_INTRINSICS_DECL__ signed char __ldcg(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (signed char)ret; }
__SM_32_INTRINSICS_DECL__ short __ldcg(const short *ptr) { unsigned short ret; asm volatile ("ld.global.cg.s16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return (short)ret; }
__SM_32_INTRINSICS_DECL__ int __ldcg(const int *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (int)ret; }
__SM_32_INTRINSICS_DECL__ long long __ldcg(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cg.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long long)ret; }
__SM_32_INTRINSICS_DECL__ char2 __ldcg(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.cg.v2.s8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ char4 __ldcg(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.cg.v4.s8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ short2 __ldcg(const short2 *ptr) { short2 ret; asm volatile ("ld.global.cg.v2.s16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ short4 __ldcg(const short4 *ptr) { short4 ret; asm volatile ("ld.global.cg.v4.s16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int2 __ldcg(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cg.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int4 __ldcg(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ longlong2 __ldcg(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.cg.v2.s64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ unsigned char __ldcg(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.u8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr));  return (unsigned char)ret; }
__SM_32_INTRINSICS_DECL__ unsigned short __ldcg(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.cg.u16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned int __ldcg(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.cg.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned long long __ldcg(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cg.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uchar2 __ldcg(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.cg.v2.u8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ uchar4 __ldcg(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.cg.v4.u8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ ushort2 __ldcg(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.cg.v2.u16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ushort4 __ldcg(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.cg.v4.u16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint2 __ldcg(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.cg.v2.u32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint4 __ldcg(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ulonglong2 __ldcg(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.cg.v2.u64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ float __ldcg(const float *ptr) { float ret; asm volatile ("ld.global.cg.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double __ldcg(const double *ptr) { double ret; asm volatile ("ld.global.cg.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float2 __ldcg(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cg.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float4 __ldcg(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double2 __ldcg(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cg.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }

/******************************************************************************
 *                                   __ldca                                    *
 ******************************************************************************/

// Size of long is architecture and OS specific.
#if defined(__LP64__) // 64 bits
__SM_32_INTRINSICS_DECL__ long __ldca(const long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldca(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
#else // 32 bits
__SM_32_INTRINSICS_DECL__ long __ldca(const long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldca(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
#endif


__SM_32_INTRINSICS_DECL__ char __ldca(const char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (char)ret; }
__SM_32_INTRINSICS_DECL__ signed char __ldca(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (signed char)ret; }
__SM_32_INTRINSICS_DECL__ short __ldca(const short *ptr) { unsigned short ret; asm volatile ("ld.global.ca.s16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return (short)ret; }
__SM_32_INTRINSICS_DECL__ int __ldca(const int *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (int)ret; }
__SM_32_INTRINSICS_DECL__ long long __ldca(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.ca.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long long)ret; }
__SM_32_INTRINSICS_DECL__ char2 __ldca(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.ca.v2.s8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ char4 __ldca(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.ca.v4.s8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ short2 __ldca(const short2 *ptr) { short2 ret; asm volatile ("ld.global.ca.v2.s16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ short4 __ldca(const short4 *ptr) { short4 ret; asm volatile ("ld.global.ca.v4.s16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int2 __ldca(const int2 *ptr) { int2 ret; asm volatile ("ld.global.ca.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int4 __ldca(const int4 *ptr) { int4 ret; asm volatile ("ld.global.ca.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ longlong2 __ldca(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.ca.v2.s64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ unsigned char __ldca(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.u8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr));  return (unsigned char)ret; }
__SM_32_INTRINSICS_DECL__ unsigned short __ldca(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.ca.u16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned int __ldca(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.ca.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned long long __ldca(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.ca.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uchar2 __ldca(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.ca.v2.u8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ uchar4 __ldca(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.ca.v4.u8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ ushort2 __ldca(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.ca.v2.u16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ushort4 __ldca(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.ca.v4.u16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint2 __ldca(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.ca.v2.u32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint4 __ldca(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ulonglong2 __ldca(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.ca.v2.u64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ float __ldca(const float *ptr) { float ret; asm volatile ("ld.global.ca.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double __ldca(const double *ptr) { double ret; asm volatile ("ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float2 __ldca(const float2 *ptr) { float2 ret; asm volatile ("ld.global.ca.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float4 __ldca(const float4 *ptr) { float4 ret; asm volatile ("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double2 __ldca(const double2 *ptr) { double2 ret; asm volatile ("ld.global.ca.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }

/******************************************************************************
 *                                   __ldcs                                    *
 ******************************************************************************/

// Size of long is architecture and OS specific.
#if defined(__LP64__) // 64 bits
__SM_32_INTRINSICS_DECL__ long __ldcs(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldcs(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
#else // 32 bits
__SM_32_INTRINSICS_DECL__ long __ldcs(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (long)ret; }
__SM_32_INTRINSICS_DECL__ unsigned long __ldcs(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
#endif


__SM_32_INTRINSICS_DECL__ char __ldcs(const char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (char)ret; }
__SM_32_INTRINSICS_DECL__ signed char __ldcs(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (signed char)ret; }
__SM_32_INTRINSICS_DECL__ short __ldcs(const short *ptr) { unsigned short ret; asm volatile ("ld.global.cs.s16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return (short)ret; }
__SM_32_INTRINSICS_DECL__ int __ldcs(const int *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return (int)ret; }
__SM_32_INTRINSICS_DECL__ long long __ldcs(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cs.s64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return (long long)ret; }
__SM_32_INTRINSICS_DECL__ char2 __ldcs(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.cs.v2.s8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ char4 __ldcs(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.cs.v4.s8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ short2 __ldcs(const short2 *ptr) { short2 ret; asm volatile ("ld.global.cs.v2.s16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ short4 __ldcs(const short4 *ptr) { short4 ret; asm volatile ("ld.global.cs.v4.s16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int2 __ldcs(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cs.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ int4 __ldcs(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ longlong2 __ldcs(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.cs.v2.s64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ unsigned char __ldcs(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.u8 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr));  return (unsigned char)ret; }
__SM_32_INTRINSICS_DECL__ unsigned short __ldcs(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.cs.u16 %0, [%1];"  : "=h"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned int __ldcs(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.cs.u32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ unsigned long long __ldcs(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cs.u64 %0, [%1];"  : "=l"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uchar2 __ldcs(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.cs.v2.u8 {%0,%1}, [%2];"  : "=r"(tmp.x), "=r"(tmp.y) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
__SM_32_INTRINSICS_DECL__ uchar4 __ldcs(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.cs.v4.u8 {%0,%1,%2,%3}, [%4];"  : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : __LDG_PTR (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
__SM_32_INTRINSICS_DECL__ ushort2 __ldcs(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.cs.v2.u16 {%0,%1}, [%2];"  : "=h"(ret.x), "=h"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ushort4 __ldcs(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.cs.v4.u16 {%0,%1,%2,%3}, [%4];"  : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint2 __ldcs(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.cs.v2.u32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ uint4 __ldcs(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ ulonglong2 __ldcs(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.cs.v2.u64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

__SM_32_INTRINSICS_DECL__ float __ldcs(const float *ptr) { float ret; asm volatile ("ld.global.cs.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double __ldcs(const double *ptr) { double ret; asm volatile ("ld.global.cs.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float2 __ldcs(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cs.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ float4 __ldcs(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
__SM_32_INTRINSICS_DECL__ double2 __ldcs(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cs.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }


#undef __LDG_PTR


// SHF is the "funnel shift" operation - an accelerated left/right shift with carry
// operating on 64-bit quantities, which are concatenations of two 32-bit registers.

// This shifts [b:a] left by "shift" bits, returning the most significant bits of the result.
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}

// This shifts [b:a] right by "shift" bits, returning the least significant bits of the result.
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}


#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 320 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_32_INTRINSICS_DECL__

#endif /* !__SM_32_INTRINSICS_HPP__ */

