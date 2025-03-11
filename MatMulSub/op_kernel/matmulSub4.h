#ifndef __MATMUL_SUB4_H_
#define __MATMUL_SUB4_H_
#include "kernel_operator.h"
#include "matmulSub4.h"
#include "lib/matmul_intf.h"

using namespace matmul;
using namespace AscendC;

template<typename T> class MatMulSub4 {
public:
    __aicore__ inline MatMulSub4() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, TPipe* pipeIn) 
    {
        blockIdx = GetBlockIdx();
        x1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1), 4194304);
        x2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2), 4194304);
        x3Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x3), 4194304);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), 4194304);
        if ASCEND_IS_AIV{
            pipeIn->InitBuffer(InQueueX3, 2, 97792);
            offset = blockIdx * 104858;
            if (blockIdx >= 24) {
                offset = blockIdx * 104857 + 24;
                tailLen = 7065;
                alignTailLen = 7;
            }
            for (int32_t i = 0; i < 5; i++) {
                if (i == 4) {
                    len = tailLen;
                    lenSize = tailLen * 4;
                    tianchong = alignTailLen;
                }
                x3LocalTensor = InQueueX3.AllocTensor<T>();
                DataCopyPad(x3LocalTensor, x3Global[offset],
                            {1, lenSize, 0, 0, 0},
                            {true, 0, tianchong, 0});
                InQueueX3.EnQue<QuePosition::GM, QuePosition::VECIN, T>(x3LocalTensor);
                x3LocalTensor = InQueueX3.DeQue<QuePosition::GM, QuePosition::VECIN, T>();
                Muls(x3LocalTensor, x3LocalTensor, (T)-1, len);
                InQueueX3.EnQue<QuePosition::VECOUT, QuePosition::GM, T>(x3LocalTensor);
                x3LocalTensor = InQueueX3.DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
                DataCopyPad(yGlobal[offset], x3LocalTensor,
                            {1, lenSize, 0, 0, 0});
                InQueueX3.FreeTensor(x3LocalTensor);
                offset += len;
            }
        }
        
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
        CrossCoreWaitFlag(0x8);

        if ASCEND_IS_AIC{
            mCoreIndx = blockIdx % 10;
            if (mCoreIndx == 9) {
                matmulsub4.SetTail(176, 1024);
            }
            offsetA = mCoreIndx * 425984;
            offsetB = blockIdx / 10 * 1024;
            offsetC = offsetA + offsetB;
            matmulsub4.SetTensorA(x1Global[offsetA]);
            matmulsub4.SetTensorB(x2Global[offsetB]);
            matmulsub4.IterateAll(yGlobal[offsetC], 1);
            matmulsub4.End();
        }
    }


public:

    typedef MatmulType<TPosition::GM, CubeFormat::ND, T> a;
    MatmulImpl<a, a, a, a, CFG_MDL> matmulsub4;

private:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> InQueueX3;
    GlobalTensor<T> x1Global; 
    GlobalTensor<T> x2Global;
    GlobalTensor<T> x3Global;
    GlobalTensor<T> yGlobal;
    LocalTensor<T> x3LocalTensor;

    int32_t blockIdx, mCoreIndx, offsetA, offsetB, offsetC;
    uint32_t tailLen = 7066, len = 24448, lenSize = 97792;
    uint8_t alignTailLen = 6, tianchong = 0;
    uint32_t offset;
};

#endif