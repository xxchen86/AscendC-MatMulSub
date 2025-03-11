#ifndef __MATMUL_SUB_FP32_H_
#define __MATMUL_SUB_FP32_H_
#include "kernel_operator.h"
#include "matmulSub_fp32.h"
#include "lib/matmul_intf.h"

using namespace matmul;
using namespace AscendC;

template<typename T> class MatMulSub_fp32 {
public:
    __aicore__ inline MatMulSub_fp32() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, const TCubeTiling &tiling, TPipe* pipeIn) 
    {
        this->tiling = tiling;
        x1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1), tiling.M * tiling.Ka);
        x2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2), tiling.Kb * tiling.N);
        x3Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x3), tiling.N);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), tiling.M * tiling.N);
        alignSingleCoreN = (tiling.singleCoreN + 7) / 8 * 8;
        pipeIn->InitBuffer(InQueueX3, 2, alignSingleCoreN * sizeof(T));
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
            x3LocalTensor = InQueueX3.AllocTensor<T>();
            DataCopyPad(x3LocalTensor, x3Global,
                        {1, (uint32_t)(tiling.singleCoreN * sizeof(T)), 0, 0, 0},
                        {true, 0, (uint8_t)(alignSingleCoreN - tiling.singleCoreN), 0});
            InQueueX3.EnQue(x3LocalTensor);
            x3LocalTensor = InQueueX3.DeQue<T>();
            Muls(x3LocalTensor, x3LocalTensor, (T)(-1), alignSingleCoreN);

            matmulsubfp32.SetTensorA(x1Global);
            matmulsubfp32.SetTensorB(x2Global);
            matmulsubfp32.SetBias(x3LocalTensor);
            if (nCoreIndx == notTailNCoreCount && mCoreIndx == notTailMCoreCount) {
                matmulsubfp32.SetTail(tailSingleCoreM, tailSingleCoreN);
            } else if (nCoreIndx == notTailNCoreCount) {
                matmulsubfp32.SetTail(tiling.singleCoreM, tailSingleCoreN);
            } else if (mCoreIndx == notTailMCoreCount) {
                matmulsubfp32.SetTail(tailSingleCoreM, tiling.singleCoreN);
            }
            matmulsubfp32.IterateAll(yGlobal);
            matmulsubfp32.End();
            InQueueX3.FreeTensor(x3LocalTensor);
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

    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> a;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> b;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> c;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> bias;
    Matmul<a, b, c, bias> matmulsubfp32;

private:
    TQue<QuePosition::VECIN, 2> InQueueX3;
    GlobalTensor<T> x1Global; 
    GlobalTensor<T> x2Global;
    GlobalTensor<T> x3Global;
    GlobalTensor<T> yGlobal;
    LocalTensor<T> x3LocalTensor;

    TCubeTiling tiling;
    int32_t mCoreIndx = 0, nCoreIndx = 0, alignSingleCoreN;
    int32_t tailSingleCoreN = 0, notTailNCoreCount = 0;
    int32_t tailSingleCoreM = 0, notTailMCoreCount = 0;
};

#endif