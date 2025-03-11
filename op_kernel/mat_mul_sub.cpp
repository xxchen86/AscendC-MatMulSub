#include "kernel_operator.h"
#include "matmulSub_fp16.h"
#include "matmulSub_fp32.h"
#include "matmulSub2.h"
#include "matmulSub4.h"

extern "C" __global__ __aicore__ void mat_mul_sub(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(5)) {
        MatMulSub4<float> matmulsub;
        matmulsub.matmulsub4.Init(&tiling_data.multiCoreTilingData);
        matmulsub.Init(x1, x2, x3, y, workspace, &pipe);
    } else if (TILING_KEY_IS(2)) {
        MatMulSub_fp32<float> matmulsub;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulsub.matmulsubfp32, &tiling_data.multiCoreTilingData);
        matmulsub.Init(x1, x2, x3, y, workspace, tiling_data.multiCoreTilingData, &pipe);
        matmulsub.Process();
    } else if (TILING_KEY_IS(3)) {
        MatMulSub2<half> matmulsub;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulsub.matmulObj, &tiling_data.multiCoreTilingData);
        matmulsub.Init(x1, x2, x3, y, workspace, tiling_data.multiCoreTilingData, &pipe);
        matmulsub.Process();
    } else if (TILING_KEY_IS(4)) {
        MatMulSub2<float> matmulsub;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulsub.matmulObj, &tiling_data.multiCoreTilingData);
        matmulsub.Init(x1, x2, x3, y, workspace, tiling_data.multiCoreTilingData, &pipe);
        matmulsub.Process();
    } else if (TILING_KEY_IS(1)) {
        MatMulSub_fp16<half> matmulsub;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulsub.matmulsubfp16, &tiling_data.multiCoreTilingData); 
        matmulsub.Init(x1, x2, x3, y, workspace, tiling_data.multiCoreTilingData, &pipe);
        matmulsub.Process();
    }
}