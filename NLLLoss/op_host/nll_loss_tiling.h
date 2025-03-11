
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NLLLossTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, reduction);
  TILING_DATA_FIELD_DEF(uint32_t, f);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, t);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NLLLoss, NLLLossTilingData)
}
