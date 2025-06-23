
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, ubPartDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
END_TILING_DATA_DEF; 
REGISTER_TILING_DATA_CLASS(Fill, FillTilingData)
}
