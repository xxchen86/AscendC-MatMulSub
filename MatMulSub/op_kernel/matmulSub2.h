#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;


template<typename T> class MatMulSub2 {
public:
    __aicore__ inline MatMulSub2(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace,
                                const TCubeTiling &tiling, TPipe *pipe)
    {
        this->tiling = tiling;  // copy
        x1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1), tiling.M * tiling.Ka);
        x2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2), tiling.Kb * tiling.N);
        x3Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x3), tiling.M * tiling.N);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), tiling.M * tiling.N);

        int offsetA = 0;
        int offsetB = 0;
        int offsetC = 0;
        CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC); // Calculate the gm offset based on the blockidx.

        x1Global = x1Global[offsetA];
        x2Global = x2Global[offsetB];
        x3Global = x3Global[offsetC];
        yGlobal = yGlobal[offsetC];
        pipe->InitBuffer(inq, 1, tiling.baseM * tiling.baseN * sizeof(T));
        pipe->InitBuffer(outq, 1, tiling.baseM * tiling.baseN * sizeof(T)); // Init output buffer.
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= tiling.usedCoreNum) {
            return;
        }
        matmulObj.SetTensorA(x1Global);
        matmulObj.SetTensorB(x2Global);
        
        if (nCoreIndx == notTailNCoreCount && mCoreIndx == notTailMCoreCount) {
            matmulObj.SetTail(tailSingleCoreM, tailSingleCoreN);
        } else if (nCoreIndx == notTailNCoreCount) {
            matmulObj.SetTail(tiling.singleCoreM, tailSingleCoreN);
        } else if (mCoreIndx == notTailMCoreCount) {
            matmulObj.SetTail(tailSingleCoreM, tiling.singleCoreN);
        }

        uint32_t computeRound = 0;
        while (matmulObj.template Iterate<true>()) {
            // COPY IN X3
            {
                auto x3Local = inq.AllocTensor<T>();
                const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
                const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
                uint32_t startOffset = (computeRound % roundM * tiling.baseM * tiling.N + computeRound / roundM * tiling.baseN);
                DataCopyParams copyParam = {(uint16_t)tiling.baseM, (uint16_t)(tiling.baseN * sizeof(T) / DEFAULT_C0_SIZE), 
                                            (uint16_t)((tiling.N - tiling.baseN) * sizeof(T) / DEFAULT_C0_SIZE), 0};
                DataCopy(x3Local, x3Global[startOffset], copyParam);
                inq.EnQue(x3Local);
            }

            // COPY IN C  &  VECTOR COMPUTE
            {
                auto x3Local = inq.DeQue<T>();
                auto outLocal = outq.AllocTensor<T>();
                matmulObj.template GetTensorC<true>(outLocal, false, true);
                Sub(outLocal, outLocal, x3Local, tiling.baseM * tiling.baseN);
                outq.EnQue(outLocal);
                inq.FreeTensor(x3Local);
            }

            // COPY OUT Y
            {
                auto outLocal = outq.DeQue<T>();
                const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
                const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
                uint32_t startOffset = (computeRound % roundM * tiling.baseM * tiling.N + computeRound / roundM * tiling.baseN);
                DataCopyParams copyParam = {(uint16_t)tiling.baseM, (uint16_t)(tiling.baseN * sizeof(T) / DEFAULT_C0_SIZE), 0,
                                            (uint16_t)((tiling.N - tiling.baseN) * sizeof(T) / DEFAULT_C0_SIZE)};
                DataCopy(yGlobal[startOffset], outLocal, copyParam);
                outq.FreeTensor(outLocal);
            }
            computeRound++;
        }
        matmulObj.End();
    }

    __aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                        int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto nSingleBlocks = Ceiling(tiling.N, tiling.singleCoreN);
        mCoreIndx = blockIdx % mSingleBlocks;
        nCoreIndx = blockIdx / mSingleBlocks;

        offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offsetB = nCoreIndx * tiling.singleCoreN;
        offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

        tailSingleCoreN = tiling.N - (nSingleBlocks - 1) * tiling.singleCoreN;
        notTailNCoreCount = nSingleBlocks - 1;
        tailSingleCoreM = tiling.M - (mSingleBlocks - 1) * tiling.singleCoreM;
        notTailMCoreCount = mSingleBlocks - 1;
    }

public:
    Matmul<
        MatmulType<TPosition::GM, CubeFormat::ND, T>,
        MatmulType<TPosition::GM, CubeFormat::ND, T>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, T>>
        matmulObj;

private:
    GlobalTensor<T> x1Global;
    GlobalTensor<T> x2Global;
    GlobalTensor<T> x3Global;
    GlobalTensor<T> yGlobal;
    TCubeTiling tiling;
    TQue<QuePosition::VECIN, 1> inq;  // x3
    TQue<QuePosition::VECOUT, 1> outq;    // x1 @ x2 and y
    TBuf<TPosition::VECCALC> buf;
    int32_t mCoreIndx = 0, nCoreIndx = 0;
    int32_t tailSingleCoreN = 0, notTailNCoreCount = 0;
    int32_t tailSingleCoreM = 0, notTailMCoreCount = 0;
};
