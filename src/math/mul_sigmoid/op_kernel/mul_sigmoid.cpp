#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mul_sigmoid.h"
#include <cstdint>

extern "C" __global__ __aicore__ void mul_sigmoid(GM_ADDR x1, GM_ADDR x2, GM_ADDR out_buf, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data_in, tiling);

  if (TILING_KEY_IS(1)) {
    MulSigmoid op;  
    op.init(x1, x2, out_buf, workspace, tiling_data_in);
    op.process();
  }
}