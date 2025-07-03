
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ClipByValueV2TilingData)
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubPartDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ClipByValueV2, ClipByValueV2TilingData)
}
