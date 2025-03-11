#include "kernel_operator.h"

using namespace AscendC;

#define align8(x) ((x + 7) / 8 * 8)

extern "C" __global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{

    if (TILING_KEY_IS(0))
    {
        TPipe pipe;
        GET_TILING_DATA(tiling_data, tiling);

        GlobalTensor<float> xgm;
        GlobalTensor<int32_t> targetgm;
        GlobalTensor<float> weightgm;
        GlobalTensor<float> ygm;
        GlobalTensor<float> wsgm;

        auto N = tiling_data.N;
        auto C = tiling_data.C;
        auto reduction = tiling_data.reduction;
        auto f = tiling_data.f;
        auto n = tiling_data.n;
        auto t = tiling_data.t;

        xgm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), N * C);
        targetgm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(target), N);
        weightgm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), C);
        ygm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), N);

        for (uint32_t i = 0; i < N; ++i)
        {
            auto t = targetgm.GetValue(i);
            auto x = xgm.GetValue(i * C + t);
            auto w = weightgm.GetValue(t);
            ygm.SetValue(i, -w * x);
        }

        if (reduction == 0)
        { // mean
            float fenmu = 0;
            for (uint32_t i = 0; i < N; ++i)
            {
                auto t = targetgm.GetValue(i);
                auto w = weightgm.GetValue(t);
                fenmu += w;
            }
            float loss_sum = 0;
            for (uint32_t i = 0; i < N; ++i)
            {
                loss_sum += ygm.GetValue(i);
            }
            ygm.SetValue(0, loss_sum / fenmu);
        }
        else if (reduction == 1)
        { // sum
            float loss_sum = 0;
            for (uint32_t i = 0; i < N; ++i)
            {
                loss_sum += ygm.GetValue(i);
            }
            ygm.SetValue(0, loss_sum);
        }
        else
        {
        }
    }
    else if (TILING_KEY_IS(1))
    {
        TPipe pipe;

        GlobalTensor<float> xgm;
        GlobalTensor<int32_t> targetgm;
        GlobalTensor<float> weightgm;
        GlobalTensor<float> ygm;
        GlobalTensor<float> wsgm;

        constexpr auto N = 1024;
        constexpr auto C = 1024;
        constexpr auto f = 32;
        constexpr auto n = 32;
        constexpr auto t = 0;

        auto id = GetBlockIdx();
        auto cur_n = f;
        if (t != 0 && id == n)
        { // tail core
            cur_n = t;
        }

        xgm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x) + id * f * C, cur_n * C);
        targetgm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(target) + id * f, cur_n);
        weightgm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), C);
        ygm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), 1);

        if (id == 32) {
            InitGlobalMemory(ygm, 1, (float)0);
        }

        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, (cur_n *3 + cur_n * C + C + cur_n * 2 + 8)*4);    // denseW denseX idx

        LocalTensor<float> denseWLocal, denseXLocal;
        LocalTensor<int32_t> idx;
        LocalTensor<float> wLocal, xLocal, yLocal;
        LocalTensor<int32_t> t1t2Local;
        LocalTensor<int32_t> t1Local, t2Local;

        denseWLocal = buf.Get<float>();
        denseXLocal = denseWLocal[cur_n];
        idx = denseXLocal[cur_n].ReinterpretCast<int32_t>();
        xLocal = idx[cur_n].ReinterpretCast<float>();
        wLocal = xLocal[cur_n * C];
        t1t2Local = wLocal[C].ReinterpretCast<int32_t>();
        t1Local = t1t2Local;
        t2Local = t1Local[cur_n];
        yLocal = t2Local[cur_n].ReinterpretCast<float>();


        DataCopy(t1Local, targetgm, cur_n);
        DataCopy(xLocal, xgm, cur_n*C);
        DataCopy(wLocal, weightgm, C);

        TQueSync<PIPE_MTE2, PIPE_V> mte2_to_v;
        mte2_to_v.SetFlag(0);
        mte2_to_v.WaitFlag(0);

        Muls(t1Local, t1Local, (int32_t)sizeof(float), cur_n);
        ArithProgression(idx, (int32_t)0, (int32_t)(C * sizeof(float)), cur_n);
        Add(t2Local, t1Local, idx, cur_n);
        Gather(denseXLocal, xLocal, t2Local.ReinterpretCast<uint32_t>(), 0, cur_n);
        Gather(denseWLocal, wLocal, t1Local.ReinterpretCast<uint32_t>(), 0, cur_n);
        Mul(denseWLocal, denseWLocal, denseXLocal, cur_n);
        ReduceSum(denseWLocal, denseWLocal, denseWLocal, cur_n);
        Muls(yLocal, denseWLocal, (float)-1, 1);

        TQueSync<PIPE_V, PIPE_MTE3> v_to_mte3;
        v_to_mte3.SetFlag(1);
        v_to_mte3.WaitFlag(1);

        SetAtomicAdd<float>();
        DataCopy(ygm, yLocal, 8);
        SetAtomicNone();
    }
}