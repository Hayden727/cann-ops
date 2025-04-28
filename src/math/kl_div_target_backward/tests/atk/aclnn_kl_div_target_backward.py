import torch

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# aclnn_kl_div_target_backward     
@register("aclnn_kl_div_target_backward")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu" or self.device == "npu":
            gradOutput = input_data.kwargs['grad_output']
            selfX = input_data.kwargs['self']
            target = input_data.kwargs['target']
            reduction = input_data.kwargs['reduction']
            logTarget = input_data.kwargs['log_target']
            if target.numel() == 0:
                return target
            # print(gradOutput)
            # print(selfX)
            # print(target)
            # print(reduction)
            # print(logTarget)
            compute_dtype = gradOutput.dtype
            gradTarget = gradOutput
            if compute_dtype == torch.bfloat16:
                gradOutput = gradOutput.to(torch.float)
                selfX = selfX.to(torch.float)
                target = target.to(torch.float)

            if logTarget:
                gradTarget = target + 1
                gradTarget = gradTarget - selfX
                tmp = torch.exp(target)
                gradTarget = gradTarget * tmp
                gradTarget = gradOutput * gradTarget
            else:
                tmp = torch.log(target)
                gradTarget = tmp + 1
                gradTarget = gradTarget - selfX
                gradTarget = gradOutput * gradTarget
                gradTarget = gradTarget.masked_fill(target==0, 0)

            if reduction == 1:
                gradTarget = gradTarget / target.numel()
            output = gradTarget.to(compute_dtype)
            # print(output)
        return output           
        
