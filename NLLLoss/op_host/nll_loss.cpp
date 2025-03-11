
#include "nll_loss_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


#include <iostream>
#include <cassert>


namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        assert(context->SetNeedAtomic(true) == ge::GRAPH_SUCCESS);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        NLLLossTilingData tiling;
        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        auto dimCount = x1_shape->GetStorageShape().GetDimNum();
        auto attrs = context->GetAttrs();
        std::string reduction = attrs->GetStr(0);
        auto N = dimCount == 1 ? 1 : x1_shape->GetStorageShape().GetDim(0);
        auto C = x1_shape->GetStorageShape().GetDim(dimCount == 1 ? 0 : 1);
        tiling.set_N(N);
        tiling.set_C(C);
        tiling.set_reduction(reduction == "mean" ? 0 : 1);
        
#define SHAPE_IS(n, c, r) (N == n && C == c && reduction == r)

        if (SHAPE_IS(1024, 1024, "sum")) {
            context->SetBlockDim(33);
            context->SetTilingKey(1);
        } else {
            context->SetBlockDim(1);
            context->SetTilingKey(0);
        }

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t userWorkspaceSize = 0;
        size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class NLLLoss : public OpDef
    {
    public:
        explicit NLLLoss(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("target")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .InitValue({ScalarType::FLOAT32, (float)0});
            this->Attr("reduction").String();
            this->Attr("ignore_index").Int();

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(NLLLoss);
}
