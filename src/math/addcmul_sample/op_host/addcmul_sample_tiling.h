/**
 * @file addcmul_sample_tiling.h
 */
 #ifndef ADDCMUL_SAMPLE_TILING_H
 #define ADDCMUL_SAMPLE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddcmulSampleTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  TILING_DATA_FIELD_DEF(uint32_t, input_data_length);
  TILING_DATA_FIELD_DEF(uint32_t, x1_length);
  TILING_DATA_FIELD_DEF(uint32_t, x2_length);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddcmulSample, AddcmulSampleTilingData)
}
// namespace optiling
#endif // ADDCMUL_SAMPLE_TILING_H