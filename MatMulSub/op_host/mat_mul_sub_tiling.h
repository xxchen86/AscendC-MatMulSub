#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(MatMulSubTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, multiCoreTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulSub, MatMulSubTilingData)
}
