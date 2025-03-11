#ifndef __MATMUL_SUB_FP16_H_
#define __MATMUL_SUB_FP16_H_
#include "kernel_operator.h"
#include "matmulSub_fp16.h"
#include "lib/matmul_intf.h"

using namespace matmul;
using namespace AscendC;
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template<typename T> class MatMulSub_fp16 {
public:
    __aicore__ inline MatMulSub_fp16() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, const TCubeTiling &tiling, TPipe* pipeIn) 
    {
        this->tiling = tiling;
        x1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1), tiling.M * tiling.Ka);
        x2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2), tiling.Kb * tiling.N);
        x3Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x3), tiling.N);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), tiling.M * tiling.N);

        coreOutLen = tiling.singleCoreN * tiling.singleCoreM;
        alignSingleCoreN = Ceiling(tiling.singleCoreN, 16) * 16;

        pipeIn->InitBuffer(InQueueX3, 2, alignSingleCoreN * sizeof(T));
        pipeIn->InitBuffer(OutQueue, 2, alignSingleCoreN * sizeof(T));
        
        int32_t offsetA = 0;
        int32_t offsetB = 0;
        int32_t offsetC = 0;
        int32_t offsetBias = 0;
        CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
        x1Global = x1Global[offsetA];
        x2Global = x2Global[offsetB];
        yGlobal = yGlobal[offsetC];
        x3Global = x3Global[offsetBias];
    }

    __aicore__ inline void Process() 
    {
        if (GetBlockIdx() < tiling.usedCoreNum) {
            int32_t singleM = tiling.singleCoreM;
            int32_t singleN = tiling.singleCoreN;
            x3LocalTensor = InQueueX3.AllocTensor<T>();
            DataCopyPad(x3LocalTensor, x3Global,
                        {1, (uint32_t)(tiling.singleCoreN * sizeof(T)), 0, 0, 0},
                        {true, 0, (uint8_t)(alignSingleCoreN - tiling.singleCoreN), 0});
            InQueueX3.EnQue(x3LocalTensor);
            halfLocalTensor = OutQueue.AllocTensor<T>();
            x3LocalTensor = InQueueX3.DeQue<T>();
            Muls(halfLocalTensor, x3LocalTensor, (T)(-1.0), alignSingleCoreN);
            OutQueue.EnQue(halfLocalTensor);
            halfLocalTensor = OutQueue.DeQue<T>();

            matmulsubfp16.SetTensorA(x1Global);
            matmulsubfp16.SetTensorB(x2Global);
            if (nCoreIndx == notTailNCoreCount && mCoreIndx == notTailMCoreCount) {
                matmulsubfp16.SetTail(tailSingleCoreM, tailSingleCoreN);
                singleM = tailSingleCoreM;
                singleN = tailSingleCoreN;
            } else if (nCoreIndx == notTailNCoreCount) {
                matmulsubfp16.SetTail(tiling.singleCoreM, tailSingleCoreN);
                singleN = tailSingleCoreN;
            } else if (mCoreIndx == notTailMCoreCount) {
                matmulsubfp16.SetTail(tailSingleCoreM, tiling.singleCoreN);
                singleM = tailSingleCoreM;
            }
            for (int32_t i = 0; i < singleM; i++) {
                DataCopyPad(yGlobal[i * tiling.N], halfLocalTensor,
                        {1, (uint32_t)(singleN * sizeof(T)), 0, 0, 0});
            }
            matmulsubfp16.IterateAll(yGlobal, 1);
            matmulsubfp16.End();
            InQueueX3.FreeTensor(x3LocalTensor);
            OutQueue.FreeTensor(halfLocalTensor);
        }
    }
    
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA,
                                                            int32_t &offsetB, int32_t &offsetC, int32_t &offsetBias)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto nSingleBlocks = Ceiling(tiling.N, tiling.singleCoreN);
        mCoreIndx = blockIdx % mSingleBlocks;
        nCoreIndx = blockIdx / mSingleBlocks;

        offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offsetB = nCoreIndx * tiling.singleCoreN;
        offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        offsetBias = nCoreIndx * tiling.singleCoreN;

        tailSingleCoreN = tiling.N - (nSingleBlocks - 1) * tiling.singleCoreN;
        notTailNCoreCount = nSingleBlocks - 1;
        tailSingleCoreM = tiling.M - (mSingleBlocks - 1) * tiling.singleCoreM;
        notTailMCoreCount = mSingleBlocks - 1;
    }


public:

    typedef MatmulType<TPosition::GM, CubeFormat::ND, T> a;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, T> b;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, T> c;
    Matmul<a, b, c> matmulsubfp16;

private:
    TQue<TPosition::VECIN, 1> InQueueX3;
    TQue<TPosition::VECOUT, 1> OutQueue;
    GlobalTensor<T> x1Global; 
    GlobalTensor<T> x2Global;
    GlobalTensor<T> x3Global;
    GlobalTensor<T> yGlobal;
    LocalTensor<T> x3LocalTensor;
    LocalTensor<T> halfLocalTensor;

    TCubeTiling tiling;
    int32_t mCoreIndx = 0, nCoreIndx = 0, alignSingleCoreN, coreOutLen;
    int32_t tailSingleCoreN = 0, notTailNCoreCount = 0;
    int32_t tailSingleCoreM = 0, notTailMCoreCount = 0;
};

#endif