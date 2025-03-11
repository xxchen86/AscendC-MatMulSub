
#include "mat_mul_sub_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  MatMulSubTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto x1 = context->GetInputTensor(0)->GetOriginShape();
  auto x2 = context->GetInputTensor(1)->GetOriginShape();
  auto x3 = context->GetInputTensor(2)->GetOriginShape();
  int32_t M = x1.GetDim(0);
  int32_t N = x2.GetDim(1);
  int32_t K = x1.GetDim(1);
  auto dataType = context->GetInputDesc(0)->GetDataType();
  MultiCoreMatmulTiling multiCoreTiling(ascendcPlatform);
  if (M == 2048 && N == 2048) {
    context->SetTilingKey(5);
    multiCoreTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    multiCoreTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    multiCoreTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    multiCoreTiling.SetDim(20);
    multiCoreTiling.SetOrgShape(M, N, K);
    multiCoreTiling.SetShape(M, N, K);
    multiCoreTiling.SetSingleShape(208, 1024, -1);
    multiCoreTiling.SetFixSplit(128, 256, -1);
    multiCoreTiling.SetTraverse(MatrixTraverse::FIRSTN);
    multiCoreTiling.SetBufferSpace(-1, -1, -1);
    multiCoreTiling.GetTiling(tiling.multiCoreTilingData);
    context->SetBlockDim(20);
  }else if (x3.GetDimNum() == 1 && static_cast<matmul_tiling::DataType>(dataType) == DataType::DT_FLOAT16) {
    context->SetTilingKey(1);
    multiCoreTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    multiCoreTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    multiCoreTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    multiCoreTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    multiCoreTiling.SetBias(false);
    multiCoreTiling.SetDim(40);
    multiCoreTiling.SetOrgShape(M, N, K);
    multiCoreTiling.SetShape(M, N, K);
    multiCoreTiling.SetBufferSpace(-1, -1, -1);
    multiCoreTiling.GetTiling(tiling.multiCoreTilingData);
    context->SetBlockDim(20);
  } else if (x3.GetDimNum() == 1 && static_cast<matmul_tiling::DataType>(dataType) == DataType::DT_FLOAT){
    context->SetTilingKey(2);
    multiCoreTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    multiCoreTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    multiCoreTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    multiCoreTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    multiCoreTiling.SetBias(true);
    multiCoreTiling.SetDim(40);
    multiCoreTiling.SetOrgShape(M, N, K);
    multiCoreTiling.SetShape(M, N, K);
    multiCoreTiling.SetBufferSpace(-1, -1, -1);
    multiCoreTiling.GetTiling(tiling.multiCoreTilingData);
    context->SetBlockDim(20);
  } else if (x3.GetDimNum() == 2 && static_cast<matmul_tiling::DataType>(dataType) == DataType::DT_FLOAT16){
    context->SetTilingKey(3);
    multiCoreTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    multiCoreTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    multiCoreTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    multiCoreTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    multiCoreTiling.SetBias(true);
    multiCoreTiling.SetDim(40);
    multiCoreTiling.SetOrgShape(M, N, K);
    multiCoreTiling.SetShape(M, N, K);
    multiCoreTiling.SetBufferSpace(-1, -1, -1);
    multiCoreTiling.GetTiling(tiling.multiCoreTilingData);
    context->SetBlockDim(20);
  } else if (x3.GetDimNum() == 2 && static_cast<matmul_tiling::DataType>(dataType) == DataType::DT_FLOAT){
    context->SetTilingKey(4);
    multiCoreTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    multiCoreTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    multiCoreTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    multiCoreTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    multiCoreTiling.SetBias(true);
    multiCoreTiling.SetDim(40);
    multiCoreTiling.SetOrgShape(M, N, K);
    multiCoreTiling.SetShape(M, N, K);
    multiCoreTiling.SetBufferSpace(-1, -1, -1);
    multiCoreTiling.GetTiling(tiling.multiCoreTilingData);
    context->SetBlockDim(20);
  }

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t userWorkspaceSize = 75 * 1024 * 1024;
  size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class MatMulSub : public OpDef {
public:
    explicit MatMulSub(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MatMulSub);
}
