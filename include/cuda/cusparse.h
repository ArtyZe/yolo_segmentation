/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
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
#if !defined(CUSPARSE_H_)
#define CUSPARSE_H_

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <library_types.h>
#include <stdint.h>

//##############################################################################
//# CUSPARSE VERSION INFORMATION
//##############################################################################

#define CUSPARSE_VER_MAJOR 10
#define CUSPARSE_VER_MINOR 3
#define CUSPARSE_VER_PATCH 0
#define CUSPARSE_VER_BUILD 243
#define CUSPARSE_VERSION (CUSPARSE_VER_MAJOR * 1000 + \
                          CUSPARSE_VER_MINOR *  100 + \
                          CUSPARSE_VER_PATCH)

//##############################################################################
//# MACRO
//##############################################################################

#ifndef CUSPARSEAPI
#    if defined(_WIN32)
#        define CUSPARSEAPI __stdcall
#    else
#        define CUSPARSEAPI
#    endif
#endif

//------------------------------------------------------------------------------

#if defined(ENABLE_CUSPARSE_DEPRECATED)

#if defined(__cplusplus) && __cplusplus >= 201402L
#   define CUSPARSE_DEPRECATED [[deprecated]]
#   define CUSPARSE_DEPRECATED_HINT(new_func)                                  \
        [[deprecated("please use " #new_func " instead")]]

#elif defined(_MSC_VER)
#   define CUSPARSE_DEPRECATED __declspec(deprecated)
#   define CUSPARSE_DEPRECATED_HINT(new_func)                                  \
        __declspec(deprecated("please use " #new_func " instead"))

#elif defined(__INTEL_COMPILER) || defined(__clang__) ||                       \
      (defined(__GNUC__) &&                                                    \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)))

#   define CUSPARSE_DEPRECATED __attribute__((deprecated))
#   define CUSPARSE_DEPRECATED_HINT(new_func)                                  \
        __attribute__((deprecated("please use " #new_func " instead")))

#elif defined(__GNUC__) || defined(__xlc__)
#   define CUSPARSE_DEPRECATED __attribute__((deprecated))
#   define CUSPARSE_DEPRECATED_HINT(new_func)                                  \
        __attribute__((deprecated))

#else
#   define CUSPARSE_DEPRECATED
#   define CUSPARSE_DEPRECATED_HINT(new_func)
#endif // defined(__cplusplus) && __cplusplus >= 201402L

#else // defined(ENABLE_CUSPARSE_DEPRECATED)
#   define CUSPARSE_DEPRECATED
#   define CUSPARSE_DEPRECATED_HINT(new_func)
#endif // !defined(ENABLE_CUSPARSE_DEPRECATED)
//------------------------------------------------------------------------------

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

//##############################################################################
//# OPAQUE DATA STRUCTURES
//##############################################################################

struct cusparseContext;
typedef struct cusparseContext* cusparseHandle_t;

struct cusparseMatDescr;
typedef struct cusparseMatDescr* cusparseMatDescr_t;

struct CUSPARSE_DEPRECATED cusparseSolveAnalysisInfo;
typedef struct CUSPARSE_DEPRECATED
cusparseSolveAnalysisInfo* cusparseSolveAnalysisInfo_t;

struct csrsv2Info;
typedef struct csrsv2Info* csrsv2Info_t;

struct csrsm2Info;
typedef struct csrsm2Info* csrsm2Info_t;

struct bsrsv2Info;
typedef struct bsrsv2Info* bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info* bsrsm2Info_t;

struct csric02Info;
typedef struct csric02Info* csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info* bsric02Info_t;

struct csrilu02Info;
typedef struct csrilu02Info* csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info* bsrilu02Info_t;

struct CUSPARSE_DEPRECATED cusparseHybMat;
typedef struct CUSPARSE_DEPRECATED cusparseHybMat* cusparseHybMat_t;

struct csrgemm2Info;
typedef struct csrgemm2Info* csrgemm2Info_t;

struct csru2csrInfo;
typedef struct csru2csrInfo* csru2csrInfo_t;

struct cusparseColorInfo;
typedef struct cusparseColorInfo* cusparseColorInfo_t;

struct pruneInfo;
typedef struct pruneInfo* pruneInfo_t;

//##############################################################################
//# ENUMERATORS
//##############################################################################

typedef enum {
    CUSPARSE_STATUS_SUCCESS                   = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED           = 1,
    CUSPARSE_STATUS_ALLOC_FAILED              = 2,
    CUSPARSE_STATUS_INVALID_VALUE             = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH             = 4,
    CUSPARSE_STATUS_MAPPING_ERROR             = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED          = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR            = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT                = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED             = 10
} cusparseStatus_t;

typedef enum {
    CUSPARSE_POINTER_MODE_HOST   = 0,
    CUSPARSE_POINTER_MODE_DEVICE = 1
} cusparsePointerMode_t;

typedef enum {
    CUSPARSE_ACTION_SYMBOLIC = 0,
    CUSPARSE_ACTION_NUMERIC  = 1
} cusparseAction_t;

typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL    = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC  = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN  = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} cusparseMatrixType_t;

typedef enum {
    CUSPARSE_FILL_MODE_LOWER = 0,
    CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
    CUSPARSE_DIAG_TYPE_UNIT     = 1
} cusparseDiagType_t;

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0,
    CUSPARSE_INDEX_BASE_ONE  = 1
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE       = 0,
    CUSPARSE_OPERATION_TRANSPOSE           = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum {
    CUSPARSE_DIRECTION_ROW    = 0,
    CUSPARSE_DIRECTION_COLUMN = 1
} cusparseDirection_t;

typedef enum {
    CUSPARSE_HYB_PARTITION_AUTO = 0,
    CUSPARSE_HYB_PARTITION_USER = 1,
    CUSPARSE_HYB_PARTITION_MAX = 2
} CUSPARSE_DEPRECATED cusparseHybPartition_t;

typedef enum {
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0,
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1
} cusparseSolvePolicy_t;

typedef enum {
    CUSPARSE_SIDE_LEFT  = 0,
    CUSPARSE_SIDE_RIGHT = 1
} cusparseSideMode_t;

typedef enum {
    CUSPARSE_COLOR_ALG0 = 0, // default
    CUSPARSE_COLOR_ALG1 = 1
} cusparseColorAlg_t;

typedef enum {
    CUSPARSE_ALG0           = 0, // default, naive
    CUSPARSE_ALG1           = 1, // merge path
    CUSPARSE_ALG_NAIVE      = 0,
    CUSPARSE_ALG_MERGE_PATH = 1 // merge path alias
} cusparseAlgMode_t;

//##############################################################################
//# INITILIAZATION AND MANAGMENT ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreate(cusparseHandle_t* handle);

cusparseStatus_t CUSPARSEAPI
cusparseDestroy(cusparseHandle_t handle);

cusparseStatus_t CUSPARSEAPI
cusparseGetVersion(cusparseHandle_t handle,
                   int*             version);

cusparseStatus_t CUSPARSEAPI
cusparseGetProperty(libraryPropertyType type,
                    int*                value);

const char* CUSPARSEAPI
cusparseGetErrorName(cusparseStatus_t status);

const char* CUSPARSEAPI
cusparseGetErrorString(cusparseStatus_t status);

cusparseStatus_t CUSPARSEAPI
cusparseSetStream(cusparseHandle_t handle,
                  cudaStream_t     streamId);

cusparseStatus_t CUSPARSEAPI
cusparseGetStream(cusparseHandle_t handle,
                  cudaStream_t*    streamId);

cusparseStatus_t CUSPARSEAPI
cusparseGetPointerMode(cusparseHandle_t       handle,
                       cusparsePointerMode_t* mode);

cusparseStatus_t CUSPARSEAPI
cusparseSetPointerMode(cusparseHandle_t      handle,
                       cusparsePointerMode_t mode);

//##############################################################################
//# HELPER ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateMatDescr(cusparseMatDescr_t* descrA);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyMatDescr(cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI
cusparseCopyMatDescr(cusparseMatDescr_t       dest,
                     const cusparseMatDescr_t src);

cusparseStatus_t CUSPARSEAPI
cusparseSetMatType(cusparseMatDescr_t   descrA,
                   cusparseMatrixType_t type);

cusparseMatrixType_t CUSPARSEAPI
cusparseGetMatType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI
cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                       cusparseFillMode_t fillMode);

cusparseFillMode_t CUSPARSEAPI
cusparseGetMatFillMode(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI
cusparseSetMatDiagType(cusparseMatDescr_t descrA,
                       cusparseDiagType_t diagType);

cusparseDiagType_t CUSPARSEAPI
cusparseGetMatDiagType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI
cusparseSetMatIndexBase(cusparseMatDescr_t  descrA,
                        cusparseIndexBase_t base);

cusparseIndexBase_t CUSPARSEAPI
cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t* info);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseGetLevelInfo(cusparseHandle_t            handle,
                     cusparseSolveAnalysisInfo_t info,
                     int*                        nlevels,
                     int**                       levelPtr,
                     int**                       levelInd);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrsv2Info(csrsv2Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrsv2Info(csrsv2Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsric02Info(csric02Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsric02Info(csric02Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsric02Info(bsric02Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsric02Info(bsric02Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrilu02Info(csrilu02Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrilu02Info(csrilu02Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrilu02Info(bsrilu02Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrilu02Info(bsrilu02Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrsv2Info(bsrsv2Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrsv2Info(bsrsv2Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrsm2Info(bsrsm2Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrsm2Info(bsrsm2Info_t info);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseCreateHybMat(cusparseHybMat_t* hybA);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDestroyHybMat(cusparseHybMat_t hybA);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsru2csrInfo(csru2csrInfo_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsru2csrInfo(csru2csrInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCreateColorInfo(cusparseColorInfo_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyColorInfo(cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseSetColorAlgs(cusparseColorInfo_t info,
                     cusparseColorAlg_t  alg);

cusparseStatus_t CUSPARSEAPI
cusparseGetColorAlgs(cusparseColorInfo_t info,
                     cusparseColorAlg_t* alg);

cusparseStatus_t CUSPARSEAPI
cusparseCreatePruneInfo(pruneInfo_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyPruneInfo(pruneInfo_t info);

//##############################################################################
//# SPARSE LEVEL 1 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSaxpyi(cusparseHandle_t    handle,
               int                 nnz,
               const float*        alpha,
               const float*        xVal,
               const int*          xInd,
               float*              y,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseDaxpyi(cusparseHandle_t    handle,
               int                 nnz,
               const double*       alpha,
               const double*       xVal,
               const int*          xInd,
               double*             y,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseCaxpyi(cusparseHandle_t    handle,
               int                 nnz,
               const cuComplex*    alpha,
               const cuComplex*    xVal,
               const int*          xInd,
               cuComplex*          y,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseZaxpyi(cusparseHandle_t       handle,
               int                    nnz,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* xVal,
               const int*             xInd,
               cuDoubleComplex*       y,
               cusparseIndexBase_t    idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseSpVV)
cusparseStatus_t CUSPARSEAPI
cusparseSdoti(cusparseHandle_t    handle,
              int                 nnz,
              const float*        xVal,
              const int*          xInd,
              const float*        y,
              float*              resultDevHostPtr,
              cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseSpVV)
cusparseStatus_t CUSPARSEAPI
cusparseDdoti(cusparseHandle_t    handle,
              int                 nnz,
              const double*       xVal,
              const int*          xInd,
              const double*       y,
              double*             resultDevHostPtr,
              cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseSpVV)
cusparseStatus_t CUSPARSEAPI
cusparseCdoti(cusparseHandle_t    handle,
              int                 nnz,
              const cuComplex*    xVal,
              const int*          xInd,
              const cuComplex*    y,
              cuComplex*          resultDevHostPtr,
              cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseSpVV)
cusparseStatus_t CUSPARSEAPI
cusparseZdoti(cusparseHandle_t       handle,
              int                    nnz,
              const cuDoubleComplex* xVal,
              const int*             xInd,
              const cuDoubleComplex* y,
              cuDoubleComplex*    resultDevHostPtr,
              cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseSpVV)
cusparseStatus_t CUSPARSEAPI
cusparseCdotci(cusparseHandle_t handle,
               int              nnz,
               const cuComplex* xVal,
               const int*       xInd,
               const cuComplex* y,
               cuComplex*       resultDevHostPtr,
               cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseSpVV)
cusparseStatus_t CUSPARSEAPI
cusparseZdotci(cusparseHandle_t       handle,
               int                    nnz,
               const cuDoubleComplex* xVal,
               const int*             xInd,
               const cuDoubleComplex* y,
               cuDoubleComplex* resultDevHostPtr,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseSgthr(cusparseHandle_t    handle,
              int                 nnz,
              const float*        y,
              float*              xVal,
              const int*          xInd,
              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseDgthr(cusparseHandle_t    handle,
              int                 nnz,
              const double*       y,
              double*             xVal,
              const int*          xInd,
              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseCgthr(cusparseHandle_t    handle,
              int                 nnz,
              const cuComplex*    y,
              cuComplex*          xVal,
              const int*          xInd,
              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseZgthr(cusparseHandle_t       handle,
              int                    nnz,
              const cuDoubleComplex* y,
              cuDoubleComplex*       xVal,
              const int*             xInd,
              cusparseIndexBase_t    idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseSgthrz(cusparseHandle_t    handle,
               int                 nnz,
               float*              y,
               float*              xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseDgthrz(cusparseHandle_t    handle,
               int                 nnz,
               double*             y,
               double*             xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseCgthrz(cusparseHandle_t    handle,
               int                 nnz,
               cuComplex*          y,
               cuComplex*          xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseZgthrz(cusparseHandle_t    handle,
               int                 nnz,
               cuDoubleComplex*    y,
               cuDoubleComplex*    xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseSsctr(cusparseHandle_t    handle,
              int                 nnz,
              const float*        xVal,
              const int*          xInd,
              float*              y,
              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseDsctr(cusparseHandle_t    handle,
              int                 nnz,
              const double*       xVal,
              const int*          xInd,
              double*             y,
              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseCsctr(cusparseHandle_t    handle,
              int                 nnz,
              const cuComplex*    xVal,
              const int*          xInd,
              cuComplex*          y,
              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseZsctr(cusparseHandle_t       handle,
              int                    nnz,
              const cuDoubleComplex* xVal,
              const int*             xInd,
              cuDoubleComplex*       y,
              cusparseIndexBase_t    idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseSroti(cusparseHandle_t    handle,
              int                 nnz,
              float*              xVal,
              const int*          xInd,
              float*              y,
              const float*        c,
              const float*        s,
              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseDroti(cusparseHandle_t    handle,
              int                 nnz,
              double*             xVal,
              const int*          xInd,
              double*             y,
              const double*       c,
              const double*       s,
              cusparseIndexBase_t idxBase);

//##############################################################################
//# SPARSE LEVEL 2 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSgemvi(cusparseHandle_t    handle,
               cusparseOperation_t transA,
               int                 m,
               int                 n,
               const float*        alpha,
               const float*        A,
               int                 lda,
               int                 nnz,
               const float*        xVal,
               const int*          xInd,
               const float*        beta,
               float*              y,
               cusparseIndexBase_t idxBase,
               void*               pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDgemvi(cusparseHandle_t    handle,
               cusparseOperation_t transA,
               int                 m,
               int                 n,
               const double*       alpha,
               const double*       A,
               int                 lda,
               int                 nnz,
               const double*       xVal,
               const int*          xInd,
               const double*       beta,
               double*             y,
               cusparseIndexBase_t idxBase,
               void*               pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCgemvi(cusparseHandle_t    handle,
               cusparseOperation_t transA,
               int                 m,
               int                 n,
               const cuComplex*    alpha,
               const cuComplex*    A,
               int                 lda,
               int                 nnz,
               const cuComplex*    xVal,
               const int*          xInd,
               const cuComplex*    beta,
               cuComplex*          y,
               cusparseIndexBase_t idxBase,
               void*               pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZgemvi(cusparseHandle_t       handle,
               cusparseOperation_t    transA,
               int                    m,
               int                    n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int                    lda,
               int                    nnz,
               const cuDoubleComplex* xVal,
               const int*             xInd,
               const cuDoubleComplex* beta,
               cuDoubleComplex*       y,
               cusparseIndexBase_t    idxBase,
               void*                  pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

CUSPARSE_DEPRECATED_HINT(cusparseSpMV)
cusparseStatus_t CUSPARSEAPI
cusparseScsrmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      nnz,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float*             csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const float*             x,
               const float*             beta,
               float*                   y);

CUSPARSE_DEPRECATED_HINT(cusparseSpMV)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      nnz,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double*            csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const double*            x,
               const double*            beta,
               double*                  y);

CUSPARSE_DEPRECATED_HINT(cusparseSpMV)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      nnz,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex*         csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const cuComplex*         x,
               const cuComplex*         beta,
               cuComplex*               y);

CUSPARSE_DEPRECATED_HINT(cusparseSpMV)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      nnz,
               const cuDoubleComplex*   alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex*   csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const cuDoubleComplex*   x,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         y);

cusparseStatus_t CUSPARSEAPI
cusparseCsrmvEx_bufferSize(cusparseHandle_t         handle,
                           cusparseAlgMode_t        alg,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      n,
                           int                      nnz,
                           const void*              alpha,
                           cudaDataType             alphatype,
                           const cusparseMatDescr_t descrA,
                           const void*              csrValA,
                           cudaDataType             csrValAtype,
                           const int*               csrRowPtrA,
                           const int*               csrColIndA,
                           const void*              x,
                           cudaDataType             xtype,
                           const void*              beta,
                           cudaDataType             betatype,
                           void*                    y,
                           cudaDataType             ytype,
                           cudaDataType             executiontype,
                           size_t*                  bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCsrmvEx(cusparseHandle_t         handle,
                cusparseAlgMode_t        alg,
                cusparseOperation_t      transA,
                int                      m,
                int                      n,
                int                      nnz,
                const void*              alpha,
                cudaDataType             alphatype,
                const cusparseMatDescr_t descrA,
                const void*              csrValA,
                cudaDataType             csrValAtype,
                const int*               csrRowPtrA,
                const int*               csrColIndA,
                const void*              x,
                cudaDataType             xtype,
                const void*              beta,
                cudaDataType             betatype,
                void*                    y,
                cudaDataType             ytype,
                cudaDataType             executiontype,
                void*                    buffer);

CUSPARSE_DEPRECATED_HINT(cusparseCsrmvEx)
cusparseStatus_t CUSPARSEAPI
cusparseScsrmv_mp(cusparseHandle_t         handle,
                  cusparseOperation_t      transA,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const float*             alpha,
                  const cusparseMatDescr_t descrA,
                  const float*             csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const float*             x,
                  const float*             beta,
                  float*                   y);

CUSPARSE_DEPRECATED_HINT(cusparseCsrmvEx)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrmv_mp(cusparseHandle_t         handle,
                  cusparseOperation_t      transA,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const double*            alpha,
                  const cusparseMatDescr_t descrA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const double*            x,
                  const double*            beta,
                  double*                  y);

CUSPARSE_DEPRECATED_HINT(cusparseCsrmvEx)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrmv_mp(cusparseHandle_t         handle,
                  cusparseOperation_t      transA,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cuComplex*         alpha,
                  const cusparseMatDescr_t descrA,
                  const cuComplex*         csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuComplex*         x,
                  const cuComplex*         beta,
                  cuComplex*               y);

CUSPARSE_DEPRECATED_HINT(cusparseCsrmvEx)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrmv_mp(cusparseHandle_t         handle,
                  cusparseOperation_t      transA,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cuDoubleComplex*   alpha,
                  const cusparseMatDescr_t descrA,
                  const cuDoubleComplex*   csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuDoubleComplex*   x,
                  const cuDoubleComplex*   beta,
                  cuDoubleComplex*         y);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseShybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const float*             x,
               const float*             beta,
               float*                   y);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDhybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const double*            x,
               const double*            beta,
               double*                  y);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseChybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const cuComplex*         x,
               const cuComplex*         beta,
               cuComplex*               y);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZhybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const cuDoubleComplex*   alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const cuDoubleComplex*   x,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         y);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float*             bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const float*             x,
               const float*             beta,
               float*                   y);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double*            bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const double*            x,
               const double*            beta,
               double*                  y);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex*         bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuComplex*         x,
               const cuComplex*         beta,
               cuComplex*               y);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuDoubleComplex*   alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex*   bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuDoubleComplex*   x,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         y);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrxmv(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const float*             alpha,
                const cusparseMatDescr_t descrA,
                const float*             bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const float*             x,
                const float*             beta,
                float*                   y);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrxmv(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const double*            alpha,
                const cusparseMatDescr_t descrA,
                const double*            bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const double*            x,
                const double*            beta,
                double*                  y);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrxmv(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const cuComplex*         alpha,
                const cusparseMatDescr_t descrA,
                const cuComplex*         bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const cuComplex*         x,
                const cuComplex*         beta,
                cuComplex*               y);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrxmv(cusparseHandle_t      handle,
             cusparseDirection_t      dirA,
             cusparseOperation_t      transA,
             int                      sizeOfMask,
             int                      mb,
             int                      nb,
             int                      nnzb,
             const cuDoubleComplex*   alpha,
             const cusparseMatDescr_t descrA,
             const cuDoubleComplex*   bsrSortedValA,
             const int*               bsrSortedMaskPtrA,
             const int*               bsrSortedRowPtrA,
             const int*               bsrSortedEndPtrA,
             const int*               bsrSortedColIndA,
             int                      blockDim,
             const cuDoubleComplex*   x,
             const cuDoubleComplex*   beta,
             cuDoubleComplex*         y);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseCsrsv_analysisEx(cusparseHandle_t            handle,
                         cusparseOperation_t         transA,
                         int                         m,
                         int                         nnz,
                         const cusparseMatDescr_t    descrA,
                         const void*                 csrSortedValA,
                         cudaDataType                csrSortedValAtype,
                         const int*                  csrSortedRowPtrA,
                         const int*                  csrSortedColIndA,
                         cusparseSolveAnalysisInfo_t info,
                         cudaDataType                executiontype);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseScsrsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const float*                csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const double*               csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const cuComplex*            csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const cuDoubleComplex*      csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseCsrsv_solveEx(cusparseHandle_t            handle,
                      cusparseOperation_t         transA,
                      int                         m,
                      const void*                 alpha,
                      cudaDataType                alphatype,
                      const cusparseMatDescr_t    descrA,
                      const void*                 csrSortedValA,
                      cudaDataType                csrSortedValAtype,
                      const int*                  csrSortedRowPtrA,
                      const int*                  csrSortedColIndA,
                      cusparseSolveAnalysisInfo_t info,
                      const void*                 f,
                      cudaDataType                ftype,
                      void*                       x,
                      cudaDataType                xtype,
                      cudaDataType                executiontype);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseScsrsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     const float*                alpha,
                     const cusparseMatDescr_t    descrA,
                     const float*                csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const float*                f,
                     float*                      x);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     const double*               alpha,
                     const cusparseMatDescr_t    descrA,
                     const double*               csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const double*               f,
                     double*                     x);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     const cuComplex*            alpha,
                     const cusparseMatDescr_t    descrA,
                     const cuComplex*            csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuComplex*            f,
                     cuComplex*                  x);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsv2)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     const cuDoubleComplex*      alpha,
                     const cusparseMatDescr_t    descrA,
                     const cuDoubleComplex*      csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuDoubleComplex*      f,
                     cuDoubleComplex*            x);

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                          csrsv2Info_t     info,
                          int*             position);

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           float*                   csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           double*                  csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              float*                   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              double*                  csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              cuComplex*               csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const double*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const float*             f,
                      float*                   x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const double*            f,
                      double*                  x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const cuComplex*         f,
                      cuComplex*               x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const cuDoubleComplex*   f,
                      cuDoubleComplex*         x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle,
                          bsrsv2Info_t     info,
                          int*             position);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           float*                   bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           double*                  bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              float*                   bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              double*                  bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuComplex*               bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex*         bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const float*             bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const double*            bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const float*             f,
                      float*                   x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const double*            f,
                      double*                  x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const cuComplex*         f,
                      cuComplex*               x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const cuDoubleComplex*   f,
                      cuDoubleComplex*         x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseShybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDhybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseChybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZhybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseShybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const float*                alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const float*                f,
                     float*                      x);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseChybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const cuComplex*            alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuComplex*            f,
                     cuComplex*                  x);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDhybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const double*               alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const double*               f,
                     double*                     x);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZhybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const cuDoubleComplex*      alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuDoubleComplex*      f,
                     cuDoubleComplex*            x);

//##############################################################################
//# SPARSE LEVEL 3 ROUTINES
//##############################################################################

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseScsrmm(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      k,
               int                      nnz,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float*             csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const float*             B,
               int                      ldb,
               const float*             beta,
               float*                   C,
               int                      ldc);

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrmm(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      k,
               int                      nnz,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double*            csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const double*            B,
               int                      ldb,
               const double*            beta,
               double*                  C,
               int                      ldc);

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrmm(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      k,
               int                      nnz,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex*         csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const cuComplex*         B,
               int                      ldb,
               const cuComplex*         beta,
               cuComplex*               C,
               int                      ldc);

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrmm(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m,
               int                      n,
               int                      k,
               int                      nnz,
               const cuDoubleComplex*   alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex*   csrSortedValA,
               const int*               csrSortedRowPtrA,
               const int*               csrSortedColIndA,
               const cuDoubleComplex*   B,
               int                      ldb,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         C,
               int                      ldc);

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseScsrmm2(cusparseHandle_t         handle,
                cusparseOperation_t      transA,
                cusparseOperation_t      transB,
                int                      m,
                int                      n,
                int                      k,
                int                      nnz,
                const float*             alpha,
                const cusparseMatDescr_t descrA,
                const float*             csrSortedValA,
                const int*               csrSortedRowPtrA,
                const int*               csrSortedColIndA,
                const float*             B,
                int                      ldb,
                const float*             beta,
                float*                   C,
                int                      ldc);

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrmm2(cusparseHandle_t         handle,
                cusparseOperation_t      transA,
                cusparseOperation_t      transB,
                int                      m,
                int                      n,
                int                      k,
                int                      nnz,
                const double*            alpha,
                const cusparseMatDescr_t descrA,
                const double* csrSortedValA,
                const int*    csrSortedRowPtrA,
                const int*    csrSortedColIndA,
                const double* B,
                int           ldb,
                const double* beta,
                double*       C,
                int           ldc);

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrmm2(cusparseHandle_t         handle,
                cusparseOperation_t      transA,
                cusparseOperation_t      transB,
                int                      m,
                int                      n,
                int                      k,
                int                      nnz,
                const cuComplex*         alpha,
                const cusparseMatDescr_t descrA,
                const cuComplex* csrSortedValA,
                const int*       csrSortedRowPtrA,
                const int*       csrSortedColIndA,
                const cuComplex* B,
                int              ldb,
                const cuComplex* beta,
                cuComplex*       C,
                int              ldc);

CUSPARSE_DEPRECATED_HINT(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrmm2(cusparseHandle_t         handle,
                cusparseOperation_t      transA,
                cusparseOperation_t      transB,
                int                      m,
                int                      n,
                int                      k,
                int                      nnz,
                const cuDoubleComplex*   alpha,
                const cusparseMatDescr_t descrA,
                const cuDoubleComplex*   csrSortedValA,
                const int*               csrSortedRowPtrA,
                const int*               csrSortedColIndA,
                const cuDoubleComplex*   B,
                int                      ldb,
                const cuDoubleComplex*   beta,
                cuDoubleComplex*         C,
                int                      ldc);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float* bsrSortedValA,
               const int*   bsrSortedRowPtrA,
               const int*   bsrSortedColIndA,
               const int    blockSize,
               const float* B,
               const int    ldb,
               const float* beta,
               float*       C,
               int          ldc);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double* bsrSortedValA,
               const int*    bsrSortedRowPtrA,
               const int*    bsrSortedColIndA,
               const int     blockSize,
               const double* B,
               const int     ldb,
               const double* beta,
               double*       C,
               int           ldc);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex* bsrSortedValA,
               const int*       bsrSortedRowPtrA,
               const int*       bsrSortedColIndA,
               const int        blockSize,
               const cuComplex* B,
               const int        ldb,
               const cuComplex* beta,
               cuComplex*       C,
               int              ldc);

cusparseStatus_t CUSPARSEAPI
 cusparseZbsrmm(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                cusparseOperation_t      transB,
                int                      mb,
                int                      n,
                int                      kb,
                int                      nnzb,
                const cuDoubleComplex*   alpha,
                const cusparseMatDescr_t descrA,
                const cuDoubleComplex*   bsrSortedValA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedColIndA,
                const int                blockSize,
                const cuDoubleComplex*   B,
                const int                ldb,
                const cuDoubleComplex*   beta,
                cuDoubleComplex*         C,
                int                      ldc);

cusparseStatus_t CUSPARSEAPI
cusparseSgemmi(cusparseHandle_t handle,
               int              m,
               int              n,
               int              k,
               int              nnz,
               const float*     alpha,
               const float*     A,
               int              lda,
               const float*     cscValB,
               const int*       cscColPtrB,
               const int*       cscRowIndB,
               const float*     beta,
               float*           C,
               int              ldc);

cusparseStatus_t CUSPARSEAPI
cusparseDgemmi(cusparseHandle_t handle,
               int              m,
               int              n,
               int              k,
               int              nnz,
               const double*    alpha,
               const double*    A,
               int              lda,
               const double*    cscValB,
               const int*       cscColPtrB,
               const int*       cscRowIndB,
               const double*    beta,
               double*          C,
               int              ldc);

cusparseStatus_t CUSPARSEAPI
cusparseCgemmi(cusparseHandle_t handle,
               int              m,
               int              n,
               int              k,
               int              nnz,
               const cuComplex* alpha,
               const cuComplex* A,
               int              lda,
               const cuComplex* cscValB,
               const int*       cscColPtrB,
               const int*       cscRowIndB,
               const cuComplex* beta,
               cuComplex*       C,
               int              ldc);

cusparseStatus_t CUSPARSEAPI
cusparseZgemmi(cusparseHandle_t       handle,
               int                    m,
               int                    n,
               int                    k,
               int                    nnz,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int                    lda,
               const cuDoubleComplex* cscValB,
               const int*             cscColPtrB,
               const int*             cscRowIndB,
               const cuDoubleComplex* beta,
               cuDoubleComplex*       C,
               int                    ldc);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseScsrsm_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const float*                csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const double*               csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const cuComplex*            csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        int                         m,
                        int                         nnz,
                        const cusparseMatDescr_t    descrA,
                        const cuDoubleComplex*      csrSortedValA,
                        const int*                  csrSortedRowPtrA,
                        const int*                  csrSortedColIndA,
                        cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseScsrsm_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     int                         n,
                     const float*                alpha,
                     const cusparseMatDescr_t    descrA,
                     const float*                csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const float*                B,
                     int                         ldb,
                     float*                      X,
                     int                         ldx);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     int                         n,
                     const double*               alpha,
                     const cusparseMatDescr_t    descrA,
                     const double*               csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const double*               B,
                     int                         ldb,
                     double*                     X,
                     int                         ldx);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     int                         n,
                     const cuComplex*            alpha,
                     const cusparseMatDescr_t    descrA,
                     const cuComplex*            csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuComplex*            B,
                     int                         ldb,
                     cuComplex*                  X,
                     int                         ldx);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrsm2)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         transA,
                     int                         m,
                     int                         n,
                     const cuDoubleComplex*      alpha,
                     const cusparseMatDescr_t    descrA,
                     const cuDoubleComplex*      csrSortedValA,
                     const int*                  csrSortedRowPtrA,
                     const int*                  csrSortedColIndA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuDoubleComplex*      B,
                     int                         ldb,
                     cuDoubleComplex*            X,
                     int                         ldx);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrsm2Info(csrsm2Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrsm2Info(csrsm2Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle,
                          csrsm2Info_t     info,
                          int* position);

cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const float*             alpha,
                              const cusparseMatDescr_t descrA,
                              const float*             csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const float*             B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const double*            alpha,
                              const cusparseMatDescr_t descrA,
                              const double*            csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const double*            B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const cuComplex*         alpha,
                              const cusparseMatDescr_t descrA,
                              const cuComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const cuComplex*         B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const cuDoubleComplex*   alpha,
                              const cusparseMatDescr_t descrA,
                              const cuDoubleComplex*   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const cuDoubleComplex*   B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const float*             alpha,
                         const cusparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const float*             B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const double*            alpha,
                         const cusparseMatDescr_t descrA,
                         const double*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const double*            B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const cuComplex*         alpha,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const cuComplex*         B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const cuDoubleComplex*   alpha,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const cuDoubleComplex*   B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      float*                   B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      double*                  B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      cuComplex*               B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      cuDoubleComplex*         B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle,
                          bsrsm2Info_t     info,
                          int*             position);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           float*                   bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           double*                  bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              float*                   bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              double*                  bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuComplex*               bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex*         bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const float*             bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const double*            bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const float*             B,
                      int                      ldb,
                      float*                   X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const double*            B,
                      int                      ldb,
                      double*                  X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const cuComplex*         B,
                      int                      ldb,
                      cuComplex*               X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const cuDoubleComplex*   B,
                      int                      ldb,
                      cuDoubleComplex*         X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer);

//##############################################################################
//# PRECONDITIONERS
//##############################################################################

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseCsrilu0Ex(cusparseHandle_t            handle,
                  cusparseOperation_t         trans,
                  int                         m,
                  const cusparseMatDescr_t    descrA,
                  void*                       csrSortedValA_ValM,
                  cudaDataType                csrSortedValA_ValMtype,
                  const int*                  csrSortedRowPtrA,
                  const int*                  csrSortedColIndA,
                  cusparseSolveAnalysisInfo_t info,
                  cudaDataType                executiontype);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseScsrilu0(cusparseHandle_t           handle,
                cusparseOperation_t         trans,
                int                         m,
                const cusparseMatDescr_t    descrA,
                float*                      csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu0(cusparseHandle_t            handle,
                 cusparseOperation_t         trans,
                 int                         m,
                 const cusparseMatDescr_t    descrA,
                 double*                     csrSortedValA_ValM,
                 const int*                  csrSortedRowPtrA,
                 const int*                  csrSortedColIndA,
                 cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu0(cusparseHandle_t         handle,
              cusparseOperation_t         trans,
              int                         m,
              const cusparseMatDescr_t    descrA,
              cuComplex*                  csrSortedValA_ValM,
              const int*                  csrSortedRowPtrA,
              const int*                  csrSortedColIndA,
              cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu0(cusparseHandle_t            handle,
                 cusparseOperation_t         trans,
                 int                         m,
                 const cusparseMatDescr_t    descrA,
                 cuDoubleComplex*            csrSortedValA_ValM,
                 const int*                  csrSortedRowPtrA,
                 const int*                  csrSortedColIndA,
                 cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle,
                            csrilu02Info_t   info,
                            int*             position);

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             float*                   csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             double*                  csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             cuComplex*               csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             cuDoubleComplex*         csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                float*                   csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                double*                  csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                cuComplex*               csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                cuDoubleComplex*         csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const float*             csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const double*            csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const cuComplex*         csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const cuDoubleComplex*   csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  float*                   csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  double*                  csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val);

cusparseStatus_t CUSPARSEAPI
cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle,
                            bsrilu02Info_t   info,
                            int*             position);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             float*                   bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             double*                  bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             cuComplex*               bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             cuDoubleComplex*         bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                float*                   bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                double*                  bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                cuComplex*               bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex*         bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsrilu02Info_t           info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           float*                   bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           double*                  bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  float*                   bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  double*                  bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseScsric0(cusparseHandle_t            handle,
                cusparseOperation_t         trans,
                int                         m,
                const cusparseMatDescr_t    descrA,
                float*                      csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseDcsric0(cusparseHandle_t         handle,
                cusparseOperation_t      trans,
                int                      m,
                const cusparseMatDescr_t descrA,
                double*                  csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseCcsric0(cusparseHandle_t         handle,
                cusparseOperation_t      trans,
                int                      m,
                const cusparseMatDescr_t descrA,
                cuComplex*               csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info);

CUSPARSE_DEPRECATED_HINT(cusparseXcsric02)
cusparseStatus_t CUSPARSEAPI
cusparseZcsric0(cusparseHandle_t            handle,
                cusparseOperation_t         trans,
                int                         m,
                const cusparseMatDescr_t    descrA,
                cuDoubleComplex*            csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseXcsric02_zeroPivot(cusparseHandle_t handle,
                           csric02Info_t    info,
                           int*             position);

cusparseStatus_t CUSPARSEAPI
cusparseScsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            float*                   csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            double*                  csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            cuComplex*               csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            cuDoubleComplex*         csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseScsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               float*                   csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               double*                  csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               cuComplex*               csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex*         csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseScsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const float*             csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const cuComplex*         csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex*   csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 float*                   csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 double*                  csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 cuComplex*               csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 cuDoubleComplex*         csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                           bsric02Info_t    info,
                           int*             position);

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            float*                   bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            double*                  bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            cuComplex*               bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            cuDoubleComplex*         bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               float*                   bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               double*                  bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               cuComplex*               bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex*         bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const float*             bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const double*            bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const cuComplex*         bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex*   bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 float*                   bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 double*                  bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 cuComplex*               bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*
                      bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 cuDoubleComplex*         bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2)
cusparseStatus_t CUSPARSEAPI
cusparseSgtsv(cusparseHandle_t handle,
              int              m,
              int              n,
              const float*     dl,
              const float*     d,
              const float*     du,
              float*           B,
              int              ldb);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2)
cusparseStatus_t CUSPARSEAPI
cusparseDgtsv(cusparseHandle_t handle,
              int              m,
              int              n,
              const double*    dl,
              const double*    d,
              const double*    du,
              double*          B,
              int              ldb);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2)
cusparseStatus_t CUSPARSEAPI
cusparseCgtsv(cusparseHandle_t handle,
              int              m,
              int              n,
              const cuComplex* dl,
              const cuComplex* d,
              const cuComplex* du,
              cuComplex*       B,
              int              ldb);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2)
cusparseStatus_t CUSPARSEAPI
cusparseZgtsv(cusparseHandle_t       handle,
              int                    m,
              int                    n,
              const cuDoubleComplex* dl,
              const cuDoubleComplex* d,
              const cuDoubleComplex* du,
              cuDoubleComplex*       B,
              int                    ldb);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int              m,
                             int              n,
                             const float*     dl,
                             const float*     d,
                             const float*     du,
                             const float*     B,
                             int              ldb,
                             size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int              m,
                             int              n,
                             const double*    dl,
                             const double*    d,
                             const double*    du,
                             const double*    B,
                             int              ldb,
                             size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int              m,
                             int              n,
                             const cuComplex* dl,
                             const cuComplex* d,
                             const cuComplex* du,
                             const cuComplex* B,
                             int              ldb,
                             size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_bufferSizeExt(cusparseHandle_t       handle,
                             int                    m,
                             int                    n,
                             const cuDoubleComplex* dl,
                             const cuDoubleComplex* d,
                             const cuDoubleComplex* du,
                             const cuDoubleComplex* B,
                             int                    ldb,
                             size_t*                bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2(cusparseHandle_t handle,
               int              m,
               int              n,
               const float*     dl,
               const float*     d,
               const float*     du,
               float*           B,
               int              ldb,
               void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2(cusparseHandle_t handle,
               int              m,
               int              n,
               const double*    dl,
               const double*    d,
               const double*    du,
               double*          B,
               int              ldb,
               void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2(cusparseHandle_t handle,
               int              m,
               int              n,
               const cuComplex* dl,
               const cuComplex* d,
               const cuComplex* du,
               cuComplex*       B,
               int              ldb,
               void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2(cusparseHandle_t       handle,
               int                    m,
               int                    n,
               const cuDoubleComplex* dl,
               const cuDoubleComplex* d,
               const cuDoubleComplex* du,
               cuDoubleComplex*       B,
               int                    ldb,
               void*                  pBuffer);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2_nopivot)
cusparseStatus_t CUSPARSEAPI
cusparseSgtsv_nopivot(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      const float*     dl,
                      const float*     d,
                      const float*     du,
                      float*           B,
                      int              ldb);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2_nopivot)
cusparseStatus_t CUSPARSEAPI
cusparseDgtsv_nopivot(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      const double*    dl,
                      const double*    d,
                      const double*    du,
                      double*          B,
                      int              ldb);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2_nopivot)
cusparseStatus_t CUSPARSEAPI
cusparseCgtsv_nopivot(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      const cuComplex* dl,
                      const cuComplex* d,
                      const cuComplex* du,
                      cuComplex*       B,
                      int              ldb);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2_nopivot)
cusparseStatus_t CUSPARSEAPI
cusparseZgtsv_nopivot(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      const cuDoubleComplex* dl,
                      const cuDoubleComplex* d,
                      const cuDoubleComplex* du,
                      cuDoubleComplex*       B,
                      int                    ldb);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const float*     dl,
                                     const float*     d,
                                     const float*     du,
                                     const float*     B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const double*    dl,
                                     const double*    d,
                                     const double*    du,
                                     const double*    B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const cuComplex* dl,
                                     const cuComplex* d,
                                     const cuComplex* du,
                                     const cuComplex* B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t       handle,
                                     int                    m,
                                     int                    n,
                                     const cuDoubleComplex* dl,
                                     const cuDoubleComplex* d,
                                     const cuDoubleComplex* du,
                                     const cuDoubleComplex* B,
                                     int                    ldb,
                                     size_t*                bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_nopivot(cusparseHandle_t handle,
                       int              m,
                       int              n,
                       const float*     dl,
                       const float*     d,
                       const float*     du,
                       float*           B,
                       int              ldb,
                       void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_nopivot(cusparseHandle_t handle,
                       int              m,
                       int              n,
                       const double*    dl,
                       const double*    d,
                       const double*    du,
                       double*          B,
                       int              ldb,
                       void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_nopivot(cusparseHandle_t handle,
                       int              m,
                       int              n,
                       const cuComplex* dl,
                       const cuComplex* d,
                       const cuComplex* du,
                       cuComplex*       B,
                       int              ldb,
                       void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_nopivot(cusparseHandle_t       handle,
                       int                    m,
                       int                    n,
                       const cuDoubleComplex* dl,
                       const cuDoubleComplex* d,
                       const cuDoubleComplex* du,
                       cuDoubleComplex*       B,
                       int                    ldb,
                       void*                  pBuffer);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2StridedBatch)
cusparseStatus_t CUSPARSEAPI
cusparseSgtsvStridedBatch(cusparseHandle_t handle,
                          int              m,
                          const float*     dl,
                          const float*     d,
                          const float*     du,
                          float*           x,
                          int              batchCount,
                          int              batchStride);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2StridedBatch)
cusparseStatus_t CUSPARSEAPI
cusparseDgtsvStridedBatch(cusparseHandle_t handle,
                          int              m,
                          const double*    dl,
                          const double*    d,
                          const double*    du,
                          double*          x,
                          int              batchCount,
                          int              batchStride);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2StridedBatch)
cusparseStatus_t CUSPARSEAPI
cusparseCgtsvStridedBatch(cusparseHandle_t handle,
                          int              m,
                          const cuComplex* dl,
                          const cuComplex* d,
                          const cuComplex* du,
                          cuComplex*       x,
                          int              batchCount,
                          int              batchStride);

CUSPARSE_DEPRECATED_HINT(cusparseXgtsv2StridedBatch)
cusparseStatus_t CUSPARSEAPI
cusparseZgtsvStridedBatch(cusparseHandle_t       handle,
                          int                    m,
                          const cuDoubleComplex* dl,
                          const cuDoubleComplex* d,
                          const cuDoubleComplex* du,
                          cuDoubleComplex*       x,
                          int                    batchCount,
                          int                    batchStride);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              m,
                                         const float*     dl,
                                         const float*     d,
                                         const float*     du,
                                         const float*     x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              m,
                                         const double*    dl,
                                         const double*    d,
                                         const double*    du,
                                         const double*    x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              m,
                                         const cuComplex* dl,
                                         const cuComplex* d,
                                         const cuComplex* du,
                                         const cuComplex* x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t       handle,
                                         int                    m,
                                         const cuDoubleComplex* dl,
                                         const cuDoubleComplex* d,
                                         const cuDoubleComplex* du,
                                         const cuDoubleComplex* x,
                                         int                    batchCount,
                                         int                    batchStride,
                                         size_t* bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2StridedBatch(cusparseHandle_t handle,
                           int              m,
                           const float*     dl,
                           const float*     d,
                           const float*     du,
                           float*           x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2StridedBatch(cusparseHandle_t handle,
                           int              m,
                           const double*    dl,
                           const double*    d,
                           const double*    du,
                           double*          x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2StridedBatch(cusparseHandle_t handle,
                           int              m,
                           const cuComplex* dl,
                           const cuComplex* d,
                           const cuComplex* du,
                           cuComplex*       x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2StridedBatch(cusparseHandle_t       handle,
                           int                    m,
                           const cuDoubleComplex* dl,
                           const cuDoubleComplex* d,
                           const cuDoubleComplex* du,
                           cuDoubleComplex*       x,
                           int                    batchCount,
                           int                    batchStride,
                           void*                  pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            const float*     x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              algo,
                                         int              m,
                                         const double*    dl,
                                         const double*    d,
                                         const double*    du,
                                         const double*    x,
                                         int              batchCount,
                                         size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const cuComplex* dl,
                                            const cuComplex* d,
                                            const cuComplex* du,
                                            const cuComplex* x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t       handle,
                                            int                    algo,
                                            int                    m,
                                            const cuDoubleComplex* dl,
                                            const cuDoubleComplex* d,
                                            const cuDoubleComplex* du,
                                            const cuDoubleComplex* x,
                                            int                    batchCount,
                                            size_t*        pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              float*           dl,
                              float*           d,
                              float*           du,
                              float*           x,
                              int              batchCount,
                              void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              double*          dl,
                              double*          d,
                              double*          du,
                              double*          x,
                              int              batchCount,
                              void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuComplex*       dl,
                              cuComplex*       d,
                              cuComplex*       du,
                              cuComplex*       x,
                              int              batchCount,
                              void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuDoubleComplex* dl,
                              cuDoubleComplex* d,
                              cuDoubleComplex* du,
                              cuDoubleComplex* x,
                              int              batchCount,
                              void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const float*     ds,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            const float*     dw,
                                            const float*     x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const double*    ds,
                                            const double*    dl,
                                            const double*    d,
                                            const double*    du,
                                            const double*    dw,
                                            const double*    x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const cuComplex* ds,
                                            const cuComplex* dl,
                                            const cuComplex* d,
                                            const cuComplex* du,
                                            const cuComplex* dw,
                                            const cuComplex* x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t       handle,
                                            int                    algo,
                                            int                    m,
                                            const cuDoubleComplex* ds,
                                            const cuDoubleComplex* dl,
                                            const cuDoubleComplex* d,
                                            const cuDoubleComplex* du,
                                            const cuDoubleComplex* dw,
                                            const cuDoubleComplex* x,
                                            int                    batchCount,
                                            size_t*         pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              float*           ds,
                              float*           dl,
                              float*           d,
                              float*           du,
                              float*           dw,
                              float*           x,
                              int              batchCount,
                              void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              double*          ds,
                              double*          dl,
                              double*          d,
                              double*          du,
                              double*          dw,
                              double*          x,
                              int              batchCount,
                              void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuComplex*       ds,
                              cuComplex*       dl,
                              cuComplex*       d,
                              cuComplex*       du,
                              cuComplex*       dw,
                              cuComplex*       x,
                              int              batchCount,
                              void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuDoubleComplex* ds,
                              cuDoubleComplex* dl,
                              cuDoubleComplex* d,
                              cuDoubleComplex* du,
                              cuDoubleComplex* dw,
                              cuDoubleComplex* x,
                              int              batchCount,
                              void*            pBuffer);

//##############################################################################
//# EXTRA ROUTINES
//##############################################################################

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2)
cusparseStatus_t CUSPARSEAPI
cusparseXcsrgemmNnz(cusparseHandle_t         handle,
                    cusparseOperation_t      transA,
                    cusparseOperation_t      transB,
                    int                      m,
                    int                      n,
                    int                      k,
                    const cusparseMatDescr_t descrA,
                    const int                nnzA,
                    const int*               csrSortedRowPtrA,
                    const int*               csrSortedColIndA,
                    const cusparseMatDescr_t descrB,
                    const int                nnzB,
                    const int*               csrSortedRowPtrB,
                    const int*               csrSortedColIndB,
                    const cusparseMatDescr_t descrC,
                    int*                     csrSortedRowPtrC,
                    int*                     nnzTotalDevHostPtr);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2)
cusparseStatus_t CUSPARSEAPI
cusparseScsrgemm(cusparseHandle_t         handle,
                 cusparseOperation_t      transA,
                 cusparseOperation_t      transB,
                 int                      m,
                 int                      n,
                 int                      k,
                 const cusparseMatDescr_t descrA,
                 const int                nnzA,
                 const float*             csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cusparseMatDescr_t descrB,
                 const int                nnzB,
                 const float*             csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 float*                   csrSortedValC,
                 const int*               csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrgemm(cusparseHandle_t         handle,
                 cusparseOperation_t      transA,
                 cusparseOperation_t      transB,
                 int                      m,
                 int                      n,
                 int                      k,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const double*            csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const double*            csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 double*                  csrSortedValC,
                 const int*               csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrgemm(cusparseHandle_t         handle,
                 cusparseOperation_t      transA,
                 cusparseOperation_t      transB,
                 int                      m,
                 int                      n,
                 int                      k,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const cuComplex*         csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 const int*               csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrgemm(cusparseHandle_t         handle,
                 cusparseOperation_t      transA,
                 cusparseOperation_t      transB,
                 int                      m,
                 int                      n,
                 int                      k,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const cuDoubleComplex*   csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const cuDoubleComplex*   csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex*         csrSortedValC,
                 const int*               csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrgemm2Info(csrgemm2Info_t* info);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrgemm2Info(csrgemm2Info_t info);

cusparseStatus_t CUSPARSEAPI
cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const float*             alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const float*             beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const double*            alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const double*            beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const cuComplex*         alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cuComplex*         beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const cuDoubleComplex*   alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cuDoubleComplex*   beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseXcsrgemm2Nnz(cusparseHandle_t         handle,
                     int                      m,
                     int                      n,
                     int                      k,
                     const cusparseMatDescr_t descrA,
                     int                      nnzA,
                     const int*               csrSortedRowPtrA,
                     const int*               csrSortedColIndA,
                     const cusparseMatDescr_t descrB,
                     int                      nnzB,
                     const int*               csrSortedRowPtrB,
                     const int*               csrSortedColIndB,
                     const cusparseMatDescr_t descrD,
                     int                      nnzD,
                     const int*               csrSortedRowPtrD,
                     const int*               csrSortedColIndD,
                     const cusparseMatDescr_t descrC,
                     int*                     csrSortedRowPtrC,
                     int*                     nnzTotalDevHostPtr,
                     const csrgemm2Info_t     info,
                     void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsrgemm2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const float*             alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const float*             csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const float*             csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const float*             beta,
                  const cusparseMatDescr_t descrD,
                  int                      nnzD,
                  const float*             csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  float*                   csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgemm2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const double*            alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const double*            csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const double*            beta,
                  const cusparseMatDescr_t descrD,
                  int                      nnzD,
                  const double*            csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  double*                  csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgemm2(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      k,
                 const cuComplex*         alpha,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const cuComplex*         csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cuComplex*         beta,
                 const cusparseMatDescr_t descrD,
                 int                      nnzD,
                 const cuComplex*         csrSortedValD,
                 const int*               csrSortedRowPtrD,
                 const int*               csrSortedColIndD,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 const int*               csrSortedRowPtrC,
                 int*                     csrSortedColIndC,
                 const csrgemm2Info_t     info,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgemm2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const cuDoubleComplex*   alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuDoubleComplex*   csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuDoubleComplex*   csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cuDoubleComplex*   beta,
                  const cusparseMatDescr_t descrD,
                  int                      nnzD,
                  const cuDoubleComplex*   csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  cuDoubleComplex*         csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2)
cusparseStatus_t CUSPARSEAPI
cusparseXcsrgeamNnz(cusparseHandle_t         handle,
                    int                      m,
                    int                      n,
                    const cusparseMatDescr_t descrA,
                    int                      nnzA,
                    const int*               csrSortedRowPtrA,
                    const int*               csrSortedColIndA,
                    const cusparseMatDescr_t descrB,
                    int                      nnzB,
                    const int*               csrSortedRowPtrB,
                    const int*               csrSortedColIndB,
                    const cusparseMatDescr_t descrC,
                    int*                     csrSortedRowPtrC,
                    int*                     nnzTotalDevHostPtr);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2)
cusparseStatus_t CUSPARSEAPI
cusparseScsrgeam(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const float*             alpha,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const float*             csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const float*             beta,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const float*             csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 float*                   csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2)
cusparseStatus_t CUSPARSEAPI
cusparseDcsrgeam(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const double*            alpha,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const double*            csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const double*            beta,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const double*            csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 double*                  csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2)
cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cuComplex*         alpha,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cuComplex*         beta,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const cuComplex*         csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2)
cusparseStatus_t CUSPARSEAPI
cusparseZcsrgeam(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cuDoubleComplex*   alpha,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const cuDoubleComplex*   csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cuDoubleComplex*   beta,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const cuDoubleComplex*   csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex*         csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const float*             alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const float*             csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const float*             beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const float*             csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const float*             csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const double*            alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const double*            csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const double*            beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const double*            csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const double*            csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const cuComplex*         alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const cuComplex*         csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuComplex*         beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const cuComplex*         csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const cuComplex*         csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const cuDoubleComplex*   alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const cuDoubleComplex*   csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuDoubleComplex*   beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const cuDoubleComplex*   csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const cuDoubleComplex*   csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseXcsrgeam2Nnz(cusparseHandle_t         handle,
                     int                      m,
                     int                      n,
                     const cusparseMatDescr_t descrA,
                     int                      nnzA,
                     const int*               csrSortedRowPtrA,
                     const int*               csrSortedColIndA,
                     const cusparseMatDescr_t descrB,
                     int                      nnzB,
                     const int*               csrSortedRowPtrB,
                     const int*               csrSortedColIndB,
                     const cusparseMatDescr_t descrC,
                     int*                     csrSortedRowPtrC,
                     int*                     nnzTotalDevHostPtr,
                     void*                    workspace);

cusparseStatus_t CUSPARSEAPI
cusparseScsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const float*             alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const float*             csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const float*             beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const float*             csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  float*                   csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const double*            alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const double*            beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const double*            csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  double*                  csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const cuComplex*         alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuComplex*         csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuComplex*         beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuComplex*         csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  cuComplex*               csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const cuDoubleComplex*   alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuDoubleComplex*   csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuDoubleComplex*   beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuDoubleComplex*   csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  cuDoubleComplex*         csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

//##############################################################################
//# SPARSE MATRIX REORDERING
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseScsrcolor(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  const float*              csrSortedValA,
                  const int*                csrSortedRowPtrA,
                  const int*                csrSortedColIndA,
                  const float*              fractionToColor,
                  int*                      ncolors,
                  int*                      coloring,
                  int*                      reordering,
                  const cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseDcsrcolor(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const double*            fractionToColor,
                  int*                     ncolors,
                  int*                     coloring,
                  int*                     reordering,
                  const cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrcolor(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  const cuComplex*          csrSortedValA,
                  const int*                csrSortedRowPtrA,
                  const int*                csrSortedColIndA,
                  const float*              fractionToColor,
                  int*                      ncolors,
                  int*                      coloring,
                  int*                      reordering,
                  const cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI
cusparseZcsrcolor(cusparseHandle_t          handle,
                  int                       m,
                  int                       nnz,
                  const cusparseMatDescr_t  descrA,
                  const cuDoubleComplex*    csrSortedValA,
                  const int*                csrSortedRowPtrA,
                  const int*                csrSortedColIndA,
                  const double*             fractionToColor,
                  int*                      ncolors,
                  int*                      coloring,
                  int*                      reordering,
                  const cusparseColorInfo_t info);

//##############################################################################
//# SPARSE FORMAT CONVERSION
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const float*             A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI
cusparseDnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const double*            A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI
cusparseCnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const cuComplex*         A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI
cusparseZnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const cuDoubleComplex*   A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

//##############################################################################
//# SPARSE FORMAT CONVERSION #
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      float                    tol);

cusparseStatus_t CUSPARSEAPI
cusparseDnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      double                   tol);

cusparseStatus_t CUSPARSEAPI
cusparseCnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      cuComplex                tol);

cusparseStatus_t CUSPARSEAPI
cusparseZnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      cuDoubleComplex          tol);

cusparseStatus_t CUSPARSEAPI
cusparseScsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const float*             csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          float*                   csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          float                    tol);

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          double*                  csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          double                   tol);

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const cuComplex*         csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          cuComplex*               csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          cuComplex                tol);

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex*   csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          cuDoubleComplex*         csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          cuDoubleComplex          tol);

cusparseStatus_t CUSPARSEAPI
cusparseSdense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   float*                   csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA);

cusparseStatus_t CUSPARSEAPI
cusparseDdense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerRow,
                   double*                  csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA);

cusparseStatus_t CUSPARSEAPI
cusparseCdense2csr(cusparseHandle_t           handle,
                     int                      m,
                     int                      n,
                     const cusparseMatDescr_t descrA,
                     const cuComplex*         A,
                     int                      lda,
                     const int*               nnzPerRow,
                     cuComplex*               csrSortedValA,
                     int*                     csrSortedRowPtrA,
                     int*                     csrSortedColIndA);

cusparseStatus_t CUSPARSEAPI
cusparseZdense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cuDoubleComplex*         csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA);

cusparseStatus_t CUSPARSEAPI
cusparseScsr2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   float*                   A,
                   int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   double*                  A,
                   int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   cuComplex*               A,
                   int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2dense(cusparseHandle_t         handle,
                int                      m,
                int                      n,
                const cusparseMatDescr_t descrA,
                const cuDoubleComplex*   csrSortedValA,
                const int*               csrSortedRowPtrA,
                const int*               csrSortedColIndA,
                cuDoubleComplex*         A,
                int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseSdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerCol,
                   float*                   cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

cusparseStatus_t CUSPARSEAPI
cusparseDdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerCol,
                   double*                  cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

cusparseStatus_t CUSPARSEAPI
cusparseCdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         A,
                   int                      lda,
                   const int*               nnzPerCol,
                   cuComplex*               cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

cusparseStatus_t CUSPARSEAPI
cusparseZdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerCol,
                   cuDoubleComplex*         cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

cusparseStatus_t CUSPARSEAPI
cusparseScsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   float*                   A,
                   int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseDcsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   double*                  A,
                   int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseCcsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   cuComplex*               A,
                   int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseZcsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   cuDoubleComplex*         A,
                   int                      lda);

cusparseStatus_t CUSPARSEAPI
cusparseXcoo2csr(cusparseHandle_t    handle,
                 const int*          cooRowInd,
                 int                 nnz,
                 int                 m,
                 int*                csrSortedRowPtr,
                 cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseXcsr2coo(cusparseHandle_t    handle,
                 const int*          csrSortedRowPtr,
                 int                 nnz,
                 int                 m,
                 int*                cooRowInd,
                 cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2)
cusparseStatus_t CUSPARSEAPI
cusparseCsr2cscEx(cusparseHandle_t    handle,
                  int                 m,
                  int                 n,
                  int                 nnz,
                  const void*         csrSortedVal,
                  cudaDataType        csrSortedValtype,
                  const int*          csrSortedRowPtr,
                  const int*          csrSortedColInd,
                  void*               cscSortedVal,
                  cudaDataType        cscSortedValtype,
                  int*                cscSortedRowInd,
                  int*                cscSortedColPtr,
                  cusparseAction_t    copyValues,
                  cusparseIndexBase_t idxBase,
                  cudaDataType        executiontype);

CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2)
cusparseStatus_t CUSPARSEAPI
cusparseScsr2csc(cusparseHandle_t    handle,
                 int                 m,
                 int                 n,
                 int                 nnz,
                 const float*        csrSortedVal,
                 const int*          csrSortedRowPtr,
                 const int*          csrSortedColInd,
                 float*              cscSortedVal,
                 int*                cscSortedRowInd,
                 int*                cscSortedColPtr,
                 cusparseAction_t    copyValues,
                 cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2)
cusparseStatus_t CUSPARSEAPI
cusparseDcsr2csc(cusparseHandle_t    handle,
                 int                 m,
                 int                 n,
                 int                 nnz,
                 const double*       csrSortedVal,
                 const int*          csrSortedRowPtr,
                 const int*          csrSortedColInd,
                 double*             cscSortedVal,
                 int*                cscSortedRowInd,
                 int*                cscSortedColPtr,
                 cusparseAction_t    copyValues,
                 cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2)
cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csc(cusparseHandle_t    handle,
                 int                 m,
                 int                 n,
                 int                 nnz,
                 const cuComplex*    csrSortedVal,
                 const int*          csrSortedRowPtr,
                 const int*          csrSortedColInd,
                 cuComplex*          cscSortedVal,
                 int*                cscSortedRowInd,
                 int*                cscSortedColPtr,
                 cusparseAction_t    copyValues,
                 cusparseIndexBase_t idxBase);

CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2)
cusparseStatus_t CUSPARSEAPI
cusparseZcsr2csc(cusparseHandle_t       handle,
                 int                    m,
                 int                    n,
                 int                    nnz,
                 const cuDoubleComplex* csrSortedVal,
                 const int*             csrSortedRowPtr,
                 const int*             csrSortedColInd,
                 cuDoubleComplex*       cscSortedVal,
                 int*                   cscSortedRowInd,
                 int*                   cscSortedColPtr,
                 cusparseAction_t       copyValues,
                 cusparseIndexBase_t    idxBase);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseSdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseCdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseShyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   float*                   A,
                   int                      lda);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDhyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   double*                  A,
                   int                      lda);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseChyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   cuComplex*               A,
                   int                      lda);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZhyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   cuDoubleComplex*         A,
                   int                      lda);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseScsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const float*             csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDcsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const double*            csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseCcsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZcsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseShyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 float*                   csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDhyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 double*                  csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseChyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuComplex*               csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZhyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuDoubleComplex*         csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseScsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const float*             cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDcsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const double*            cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseCcsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZcsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseShyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 float*                   cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseDhyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 double*                  cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseChyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuComplex*               cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr);

CUSPARSE_DEPRECATED
cusparseStatus_t CUSPARSEAPI
cusparseZhyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuDoubleComplex*         cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr);

cusparseStatus_t CUSPARSEAPI
cusparseXcsr2bsrNnz(cusparseHandle_t         handle,
                    cusparseDirection_t      dirA,
                    int                      m,
                    int                      n,
                    const cusparseMatDescr_t descrA,
                    const int*               csrSortedRowPtrA,
                    const int*               csrSortedColIndA,
                    int                      blockDim,
                    const cusparseMatDescr_t descrC,
                    int*                     bsrSortedRowPtrC,
                    int*                     nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI
cusparseScsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const float*             csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 float*                   bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const double*            csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 double*                  bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex*         bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseSbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const float*             bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 float*                   csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseDbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const double*            bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 double*                  csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseCbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseZbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex*         csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const float*     bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const double*    bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const cuComplex* bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t       handle,
                                int                    mb,
                                int                    nb,
                                int                    nnzb,
                                const cuDoubleComplex* bsrSortedVal,
                                const int*             bsrSortedRowPtr,
                                const int*             bsrSortedColInd,
                                int                    rowBlockDim,
                                int                    colBlockDim,
                                int*                   pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const float*     bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const double*    bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const cuComplex* bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc_bufferSizeExt(cusparseHandle_t       handle,
                                   int                    mb,
                                   int                    nb,
                                   int                    nnzb,
                                   const cuDoubleComplex* bsrSortedVal,
                                   const int*             bsrSortedRowPtr,
                                   const int*             bsrSortedColInd,
                                   int                    rowBlockDim,
                                   int                    colBlockDim,
                                   size_t*                pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc(cusparseHandle_t handle,
                     int              mb,
                     int              nb,
                     int              nnzb,
                     const float*     bsrSortedVal,
                     const int* bsrSortedRowPtr,
                     const int* bsrSortedColInd,
                     int        rowBlockDim,
                     int        colBlockDim,
                     float*     bscVal,
                     int*       bscRowInd,
                     int*       bscColPtr,
                     cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase,
                     void*               pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc(cusparseHandle_t    handle,
                     int                 mb,
                     int                 nb,
                     int                 nnzb,
                     const double*       bsrSortedVal,
                     const int*          bsrSortedRowPtr,
                     const int*          bsrSortedColInd,
                     int                 rowBlockDim,
                     int                 colBlockDim,
                     double*             bscVal,
                     int*                bscRowInd,
                     int*                bscColPtr,
                     cusparseAction_t    copyValues,
                     cusparseIndexBase_t idxBase,
                     void*               pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc(cusparseHandle_t    handle,
                     int                 mb,
                     int                 nb,
                     int                 nnzb,
                     const cuComplex*    bsrSortedVal,
                     const int*          bsrSortedRowPtr,
                     const int*          bsrSortedColInd,
                     int                 rowBlockDim,
                     int                 colBlockDim,
                     cuComplex*          bscVal,
                     int*                bscRowInd,
                     int*                bscColPtr,
                     cusparseAction_t    copyValues,
                     cusparseIndexBase_t idxBase,
                     void*               pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc(cusparseHandle_t       handle,
                     int                    mb,
                     int                    nb,
                     int                    nnzb,
                     const cuDoubleComplex* bsrSortedVal,
                     const int*             bsrSortedRowPtr,
                     const int*             bsrSortedColInd,
                     int                    rowBlockDim,
                     int                    colBlockDim,
                     cuDoubleComplex*       bscVal,
                     int*                   bscRowInd,
                     int*                   bscColPtr,
                     cusparseAction_t       copyValues,
                     cusparseIndexBase_t    idxBase,
                     void*                  pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseXgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const float*             bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   float*                   csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const double*            bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   double*                  csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   cuComplex*               csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   cuDoubleComplex*         csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const float*             csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const double*            csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const cuComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const cuDoubleComplex*   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const float*             csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const double*            csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const cuComplex*         csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const cuDoubleComplex*   csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseXcsr2gebsrNnz(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      int                      m,
                      int                      n,
                      const cusparseMatDescr_t descrA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const cusparseMatDescr_t descrC,
                      int*                     bsrSortedRowPtrC,
                      int                      rowBlockDim,
                      int                      colBlockDim,
                      int*                     nnzTotalDevHostPtr,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   float*                   bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   double*                  bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   cuComplex*               bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   cuDoubleComplex*         bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const float*             bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const double*            bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const cuComplex*         bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const cuDoubleComplex*   bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const float*             bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const double*            bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const cuComplex*         bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const cuDoubleComplex*   bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize);

cusparseStatus_t CUSPARSEAPI
cusparseXgebsr2gebsrNnz(cusparseHandle_t         handle,
                        cusparseDirection_t      dirA,
                        int                      mb,
                        int                      nb,
                        int                      nnzb,
                        const cusparseMatDescr_t descrA,
                        const int*               bsrSortedRowPtrA,
                        const int*               bsrSortedColIndA,
                        int                      rowBlockDimA,
                        int                      colBlockDimA,
                        const cusparseMatDescr_t descrC,
                        int*                     bsrSortedRowPtrC,
                        int                      rowBlockDimC,
                        int                      colBlockDimC,
                        int*                     nnzTotalDevHostPtr,
                        void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const float*             bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     float*                   bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const double*            bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     double*                  bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const cuComplex*         bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     cuComplex*               bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const cuDoubleComplex*   bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     cuDoubleComplex*         bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

//##############################################################################
//# SPARSE MATRIX SORTING
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateIdentityPermutation(cusparseHandle_t handle,
                                  int              n,
                                  int*             p);

cusparseStatus_t CUSPARSEAPI
cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       cooRowsA,
                               const int*       cooColsA,
                               size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseXcoosortByRow(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      int              nnz,
                      int*             cooRowsA,
                      int*             cooColsA,
                      int*             P,
                      void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseXcoosortByColumn(cusparseHandle_t handle,
                         int              m,
                         int              n,
                         int              nnz,
                         int*             cooRowsA,
                         int*             cooColsA,
                         int*             P,
                         void*            pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       csrRowPtrA,
                               const int*       csrColIndA,
                               size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsort(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 const int*               csrRowPtrA,
                 int*                     csrColIndA,
                 int*                     P,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       cscColPtrA,
                               const int*       cscRowIndA,
                               size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseXcscsort(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 const int*               cscColPtrA,
                 int*                     cscRowIndA,
                 int*                     P,
                 void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                float*           csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                double*          csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuComplex*       csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuDoubleComplex* csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseScsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  float*                   csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  double*                  csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseScsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  float*                   csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  double*                  csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csr_bufferSizeExt(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const __half*            A,
                                      int                      lda,
                                      const __half*            threshold,
                                      const cusparseMatDescr_t descrC,
                                      const __half*            csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t* pBufferSizeInBytes);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csr_bufferSizeExt(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const float*             A,
                                      int                      lda,
                                      const float*             threshold,
                                      const cusparseMatDescr_t descrC,
                                      const float*             csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t* pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csr_bufferSizeExt(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const double*            A,
                                      int                      lda,
                                      const double*            threshold,
                                      const cusparseMatDescr_t descrC,
                                      const double*            csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t*               pBufferSizeInBytes);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrNnz(cusparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const __half*            A,
                           int                      lda,
                           const __half*            threshold,
                           const cusparseMatDescr_t descrC,
                           int*                     csrRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrNnz(cusparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const float*             A,
                           int                      lda,
                           const float*             threshold,
                           const cusparseMatDescr_t descrC,
                           int*                     csrRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrNnz(cusparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const double*            A,
                           int                      lda,
                           const double*            threshold,
                           const cusparseMatDescr_t descrC,
                           int*                     csrSortedRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csr(cusparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const __half*            A,
                        int                      lda,
                        const __half*            threshold,
                        const cusparseMatDescr_t descrC,
                        __half*                  csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csr(cusparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const float*             A,
                        int                      lda,
                        const float*             threshold,
                        const cusparseMatDescr_t descrC,
                        float*                   csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csr(cusparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const double*            A,
                        int                      lda,
                        const double*            threshold,
                        const cusparseMatDescr_t descrC,
                        double*                  csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csr_bufferSizeExt(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const __half*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const __half*            threshold,
                                    const cusparseMatDescr_t descrC,
                                    const __half*            csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t* pBufferSizeInBytes);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csr_bufferSizeExt(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const float*             csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const float*             threshold,
                                    const cusparseMatDescr_t descrC,
                                    const float*             csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t*                 pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csr_bufferSizeExt(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const double*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const double*            threshold,
                                    const cusparseMatDescr_t descrC,
                                    const double*            csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t*                 pBufferSizeInBytes);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrNnz(cusparseHandle_t         handle,
                         int                      m,
                         int                      n,
                         int                      nnzA,
                         const cusparseMatDescr_t descrA,
                         const __half*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const __half*            threshold,
                         const cusparseMatDescr_t descrC,
                         int*                     csrSortedRowPtrC,
                         int*                     nnzTotalDevHostPtr,
                         void*                    pBuffer);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrNnz(cusparseHandle_t         handle,
                         int                      m,
                         int                      n,
                         int                      nnzA,
                         const cusparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const float*             threshold,
                         const cusparseMatDescr_t descrC,
                         int*                     csrSortedRowPtrC,
                         int*                     nnzTotalDevHostPtr,
                         void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
 cusparseDpruneCsr2csrNnz(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          int                      nnzA,
                          const cusparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          const double*            threshold,
                          const cusparseMatDescr_t descrC,
                          int*                     csrSortedRowPtrC,
                          int*                     nnzTotalDevHostPtr,
                          void*                    pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csr(cusparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const cusparseMatDescr_t descrA,
                      const __half*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const __half*            threshold,
                      const cusparseMatDescr_t descrC,
                      __half*                  csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csr(cusparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const cusparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const float*             threshold,
                      const cusparseMatDescr_t descrC,
                      float*                   csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csr(cusparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const cusparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const double*            threshold,
                      const cusparseMatDescr_t descrC,
                      double*                  csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const __half*            A,
                                   int                      lda,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const __half*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const float*             A,
                                   int                      lda,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const float*             csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const double*            A,
                                   int                      lda,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const double*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const __half*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const float*             A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const double*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrByPercentage(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const __half*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    __half*                  csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrByPercentage(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const float*             A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    float*                   csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrByPercentage(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const double*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    double*                  csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#if defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const cusparseMatDescr_t descrA,
                                   const __half*            csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const __half*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const cusparseMatDescr_t descrA,
                                   const float*             csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const float*             csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const cusparseMatDescr_t descrA,
                                   const double*            csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const double*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

#if defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const __half*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const float*             csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const double*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrByPercentage(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const __half*            csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float percentage, /* between 0 to 100 */
                                  const cusparseMatDescr_t descrC,
                                  __half*                  csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  pruneInfo_t              info,
                                  void*                    pBuffer);

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrByPercentage(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const float*             csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float                    percentage,
                                  const cusparseMatDescr_t descrC,
                                  float*                   csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  pruneInfo_t              info,
                                  void*                    pBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrByPercentage(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const double*            csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float                    percentage,
                                  const cusparseMatDescr_t descrC,
                                  double*                  csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  pruneInfo_t              info,
                                  void*                    pBuffer);

//##############################################################################
//# CSR2CSC
//##############################################################################

typedef enum {
    CUSPARSE_CSR2CSC_ALG1 = 1, // faster than V2 (in general), deterministc
    CUSPARSE_CSR2CSC_ALG2 = 2  // low memory requirement, non-deterministc
} cusparseCsr2CscAlg_t;

cusparseStatus_t CUSPARSEAPI
cusparseCsr2cscEx2(cusparseHandle_t     handle,
                   int                  m,
                   int                  n,
                   int                  nnz,
                   const void*          csrVal,
                   const int*           csrRowPtr,
                   const int*           csrColInd,
                   void*                cscVal,
                   int*                 cscColPtr,
                   int*                 cscRowInd,
                   cudaDataType         valType,
                   cusparseAction_t     copyValues,
                   cusparseIndexBase_t  idxBase,
                   cusparseCsr2CscAlg_t alg,
                   void*                buffer);

cusparseStatus_t CUSPARSEAPI
cusparseCsr2cscEx2_bufferSize(cusparseHandle_t     handle,
                              int                  m,
                              int                  n,
                              int                  nnz,
                              const void*          csrVal,
                              const int*           csrRowPtr,
                              const int*           csrColInd,
                              void*                cscVal,
                              int*                 cscColPtr,
                              int*                 cscRowInd,
                              cudaDataType         valType,
                              cusparseAction_t     copyValues,
                              cusparseIndexBase_t  idxBase,
                              cusparseCsr2CscAlg_t alg,
                              size_t*              bufferSize);

//##############################################################################
//# SpMM APIs
//##############################################################################

#if !defined(_WIN32)

typedef enum {
    CUSPARSE_FORMAT_CSR         = 1, ///< Compressed Sparse Row (CSR)
    CUSPARSE_FORMAT_CSC         = 2, ///< Compressed Sparse Column (CSC)
    CUSPARSE_FORMAT_COO         = 3, ///< Coordinate (COO) - Structure of Arrays
    CUSPARSE_FORMAT_COO_AOS     = 4, ///< Coordinate (COO) - Array of Structures
} cusparseFormat_t;

typedef enum {
    CUSPARSE_ORDER_COL = 1, ///< Column-Major Order - Matrix memory layout
    CUSPARSE_ORDER_ROW = 2  ///< Row-Major Order - Matrix memory layout
} cusparseOrder_t;

typedef enum {
    CUSPARSE_MV_ALG_DEFAULT = 0,
    CUSPARSE_COOMV_ALG      = 1,
    CUSPARSE_CSRMV_ALG1     = 2,
    CUSPARSE_CSRMV_ALG2     = 3
} cusparseSpMVAlg_t;

typedef enum {
    CUSPARSE_MM_ALG_DEFAULT = 0,
    CUSPARSE_COOMM_ALG1 = 1, // non-deterministc results
    CUSPARSE_COOMM_ALG2 = 2, // deterministic results
    CUSPARSE_COOMM_ALG3 = 3, // non-deterministc results, for large matrices
    CUSPARSE_CSRMM_ALG1 = 4
} cusparseSpMMAlg_t;

typedef enum {
    CUSPARSE_INDEX_16U = 1, ///< 16-bit unsigned integer for matrix/vector
                            ///< indices
    CUSPARSE_INDEX_32I = 2, ///< 32-bit signed integer for matrix/vector indices
    CUSPARSE_INDEX_64I = 3  ///< 64-bit signed integer for matrix/vector indices
} cusparseIndexType_t;

//------------------------------------------------------------------------------

struct cusparseSpVecDescr;
struct cusparseDnVecDescr;
struct cusparseSpMatDescr;
struct cusparseDnMatDescr;
typedef struct cusparseSpVecDescr* cusparseSpVecDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseDnMatDescr* cusparseDnMatDescr_t;

//------------------------------------------------------------------------------
// SPARSE VECTOR DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr,
                    int64_t               size,
                    int64_t               nnz,
                    void*                 indices,
                    void*                 values,
                    cusparseIndexType_t   idxType,
                    cusparseIndexBase_t   idxBase,
                    cudaDataType          valueType);

cusparseStatus_t CUSPARSEAPI
cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr);

cusparseStatus_t CUSPARSEAPI
cusparseSpVecGet(const cusparseSpVecDescr_t spVecDescr,
                 int64_t*                   size,
                 int64_t*                   nnz,
                 void**                     indices,
                 void**                     values,
                 cusparseIndexType_t*       idxType,
                 cusparseIndexBase_t*       idxBase,
                 cudaDataType*              valueType);

cusparseStatus_t CUSPARSEAPI
cusparseSpVecGetIndexBase(const cusparseSpVecDescr_t spVecDescr,
                          cusparseIndexBase_t*       idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseSpVecGetValues(const cusparseSpVecDescr_t spVecDescr,
                       void**                     values);

cusparseStatus_t CUSPARSEAPI
cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr,
                       void*                values);

//------------------------------------------------------------------------------
// DENSE VECTOR DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                    int64_t               size,
                    void*                 values,
                    cudaDataType          valueType);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr);

cusparseStatus_t CUSPARSEAPI
cusparseDnVecGet(const cusparseDnVecDescr_t dnVecDescr,
                 int64_t*                   size,
                 void**                     values,
                 cudaDataType*              valueType);

cusparseStatus_t CUSPARSEAPI
cusparseDnVecGetValues(const cusparseDnVecDescr_t dnVecDescr,
                       void**                     values);

cusparseStatus_t CUSPARSEAPI
cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr,
                       void*                values);

//------------------------------------------------------------------------------
// SPARSE MATRIX DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 cooRowInd,
                  void*                 cooColInd,
                  void*                 cooValues,
                  cusparseIndexType_t   cooIdxType,
                  cusparseIndexBase_t   idxBase,
                  cudaDataType          valueType);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 csrRowOffsets,
                  void*                 csrColInd,
                  void*                 csrValues,
                  cusparseIndexType_t   csrRowOffsetsType,
                  cusparseIndexType_t   csrColIndType,
                  cusparseIndexBase_t   idxBase,
                  cudaDataType          valueType);

cusparseStatus_t CUSPARSEAPI
cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr,
                     int64_t               rows,
                     int64_t               cols,
                     int64_t               nnz,
                     void*                 cooInd,
                     void*                 cooValues,
                     cusparseIndexType_t   cooIdxType,
                     cusparseIndexBase_t   idxBase,
                     cudaDataType          valueType);

cusparseStatus_t CUSPARSEAPI
cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr);

cusparseStatus_t CUSPARSEAPI
cusparseCooGet(const cusparseSpMatDescr_t spMatDescr,
               int64_t*                   rows,
               int64_t*                   cols,
               int64_t*                   nnz,
               void**                     cooRowInd,  // COO row indices
               void**                     cooColInd,  // COO column indices
               void**                     cooValues,  // COO values
               cusparseIndexType_t*       idxType,
               cusparseIndexBase_t*       idxBase,
               cudaDataType*              valueType);

cusparseStatus_t CUSPARSEAPI
cusparseCooAoSGet(const cusparseSpMatDescr_t spMatDescr,
                  int64_t*                   rows,
                  int64_t*                   cols,
                  int64_t*                   nnz,
                  void**                     cooInd,     // COO indices
                  void**                     cooValues,  // COO values
                  cusparseIndexType_t*       idxType,
                  cusparseIndexBase_t*       idxBase,
                  cudaDataType*              valueType);

cusparseStatus_t CUSPARSEAPI
cusparseCsrGet(const cusparseSpMatDescr_t spMatDescr,
               int64_t*                   rows,
               int64_t*                   cols,
               int64_t*                   nnz,
               void**                     csrRowOffsets,
               void**                     csrColInd,
               void**                     csrValues,
               cusparseIndexType_t*       csrRowOffsetsType,
               cusparseIndexType_t*       csrColIndType,
               cusparseIndexBase_t*       idxBase,
               cudaDataType*              valueType);

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetFormat(const cusparseSpMatDescr_t spMatDescr,
                       cusparseFormat_t*          format);

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetIndexBase(const cusparseSpMatDescr_t spMatDescr,
                          cusparseIndexBase_t*       idxBase);

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetValues(const cusparseSpMatDescr_t spMatDescr,
                       void**                     values);

cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr,
                       void*                values);

cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                             int                  batchCount);

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetStridedBatch(const cusparseSpMatDescr_t spMatDescr,
                             int*                       batchCount);

//------------------------------------------------------------------------------
// DENSE MATRIX DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                    int64_t               rows,
                    int64_t               cols,
                    int64_t               ld,
                    void*                 values,
                    cudaDataType          valueType,
                    cusparseOrder_t       order);

cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr);

cusparseStatus_t CUSPARSEAPI
cusparseDnMatGet(const cusparseDnMatDescr_t dnMatDescr,
                 int64_t*                   rows,
                 int64_t*                   cols,
                 int64_t*                   ld,
                 void**                     values,
                 cudaDataType*              type,
                 cusparseOrder_t*           order);

cusparseStatus_t CUSPARSEAPI
cusparseDnMatGetValues(const cusparseDnMatDescr_t dnMatDescr,
                       void**                     values);

cusparseStatus_t CUSPARSEAPI
cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr,
                       void*                values);

cusparseStatus_t CUSPARSEAPI
cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                             int                  batchCount,
                             int64_t              batchStride);

cusparseStatus_t CUSPARSEAPI
cusparseDnMatGetStridedBatch(const cusparseDnMatDescr_t dnMatDescr,
                             int*                       batchCount,
                             int64_t*                   batchStride);

//------------------------------------------------------------------------------
// SPARSE VECTOR-VECTOR MULTIPLICATION

cusparseStatus_t CUSPARSEAPI
cusparseSpVV(cusparseHandle_t           handle,
             cusparseOperation_t        opX,
             const cusparseSpVecDescr_t vecX,
             const cusparseDnVecDescr_t vecY,
             void*                      result,
             cudaDataType               computeType,
             void*                      externalBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSpVV_bufferSize(cusparseHandle_t           handle,
                        cusparseOperation_t        opX,
                        const cusparseSpVecDescr_t vecX,
                        const cusparseDnVecDescr_t vecY,
                        const void*                result,
                        cudaDataType               computeType,
                        size_t*                    bufferSize);

//------------------------------------------------------------------------------
// SPARSE MATRIX-VECTOR MULTIPLICATION

cusparseStatus_t CUSPARSEAPI
cusparseSpMV(cusparseHandle_t           handle,
             cusparseOperation_t        opA,
             const void*                alpha,
             const cusparseSpMatDescr_t matA,
             const cusparseDnVecDescr_t vecX,
             const void*                beta,
             const cusparseDnVecDescr_t vecY,
             cudaDataType               computeType,
             cusparseSpMVAlg_t          alg,
             void*                      externalBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSpMV_bufferSize(cusparseHandle_t           handle,
                        cusparseOperation_t        opA,
                        const void*                alpha,
                        const cusparseSpMatDescr_t matA,
                        const cusparseDnVecDescr_t vecX,
                        const void*                beta,
                        const cusparseDnVecDescr_t vecY,
                        cudaDataType               computeType,
                        cusparseSpMVAlg_t          alg,
                        size_t*                    bufferSize);

//------------------------------------------------------------------------------
// SPARSE MATRIX-MATRIX MULTIPLICATION

cusparseStatus_t CUSPARSEAPI
cusparseSpMM(cusparseHandle_t           handle,
             cusparseOperation_t        opA,
             cusparseOperation_t        opB,
             const void*                alpha,
             const cusparseSpMatDescr_t matA,
             const cusparseDnMatDescr_t matB,
             const void*                beta,
             cusparseDnMatDescr_t       matC,
             cudaDataType               computeType,
             cusparseSpMMAlg_t          alg,
             void*                      externalBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseSpMM_bufferSize(cusparseHandle_t           handle,
                        cusparseOperation_t        opA,
                        cusparseOperation_t        opB,
                        const void*                alpha,
                        const cusparseSpMatDescr_t matA,
                        const cusparseDnMatDescr_t matB,
                        const void*                beta,
                        cusparseDnMatDescr_t       matC,
                        cudaDataType               computeType,
                        cusparseSpMMAlg_t          alg,
                        size_t*                    bufferSize);

//------------------------------------------------------------------------------
// GENERAL MATRIX-MATRIX PATTERN-CONSTRAINED MULTIPLICATION

cusparseStatus_t CUSPARSEAPI
cusparseConstrainedGeMM(cusparseHandle_t           handle,
                        cusparseOperation_t        opA,
                        cusparseOperation_t        opB,
                        const void*                alpha,
                        const cusparseDnMatDescr_t matA,
                        const cusparseDnMatDescr_t matB,
                        const void*                beta,
                        cusparseSpMatDescr_t       matC,
                        cudaDataType               computeType,
                        void*                      externalBuffer);

cusparseStatus_t CUSPARSEAPI
cusparseConstrainedGeMM_bufferSize(cusparseHandle_t           handle,
                                   cusparseOperation_t        opA,
                                   cusparseOperation_t        opB,
                                   const void*                alpha,
                                   const cusparseDnMatDescr_t matA,
                                   const cusparseDnMatDescr_t matB,
                                   const void*                beta,
                                   cusparseSpMatDescr_t       matC,
                                   cudaDataType               computeType,
                                   size_t*                    bufferSize);

#endif // !defined(_WIN32)

//==============================================================================
//==============================================================================

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#undef CUSPARSE_DEPRECATED
#undef CUSPARSE_DEPRECATED_HINT

#endif // !defined(CUSPARSE_H_)
