
#include "opdev/op_log.h"

#ifdef __cplusplus
extern "C" {
#endif

inline int64_t Ceil(int64_t x, int64_t y) {
  if (y == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
    return INT64_MIN;
  }
  return ((x + y - 1) / y) * y;
}

inline int64_t CeilDiv(int64_t x, int64_t y) {
  if (y == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
    return INT64_MIN;
  }
  return (x + y - 1) / y;
}

#ifdef __cplusplus
}
#endif
