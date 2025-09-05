#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_quant_v2(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const at::Tensor& scales1,
    const c10::optional<at::Tensor> &scales2,
    const c10::optional<at::Tensor> &zero_points1,
    const c10::optional<at::Tensor> &zero_points2,
    const c10::optional<at::Tensor> &bias,
    int64_t axis,
    double epsilon,
    bool div_mode
    )
{
    at::Tensor y1 = at::empty_like(x1, at::TensorOptions().dtype(at::kChar).device(x1.options().device()));
    at::Tensor y2 = at::empty_like(x1, at::TensorOptions().dtype(at::kChar).device(x1.options().device()));
    at::Tensor x = at::empty_like(x1, at::TensorOptions().dtype(x1.scalar_type()).device(x1.options().device()));

    EXEC_NPU_CMD(aclnnAddRmsNormQuantV2, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, bias, axis, epsilon, div_mode, y1, y2, x);
    return std::make_tuple(y1, y2, x);
}
}
