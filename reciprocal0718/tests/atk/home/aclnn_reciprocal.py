import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("aclnn_cpu_reciprocal")
class FunctionApi(BaseApi):
    # def __call__(self, input_data: InputDataset, with_output: bool = False):
    #     if self.device == "cpu" or self.device == "npu":
    #         output = torch.reciprocal(
    #             *input_data.args, **input_data.kwargs
    #         )
    #     return output

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if not with_output:
            eval(self.api_name)(*input_data.args, **input_data.kwargs)
            return
        if self.output is None:
            output = eval(self.api_name)(
                *input_data.args, **input_data.kwargs
            )
        else:
            eval(self.api_name)(*input_data.args, **input_data.kwargs)
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        return output
    def get_format(self, input_data: InputDataset, index=None, name=None):
            """
            :param input_data: 参数列表
            :param index: 参数位置
            :param name: 参数名字
            :return:
            format at this index or name
            """
            if input_data.kwargs["format"] == "NCHW":
                return AclFormat.ACL_FORMAT_NCHW
            if input_data.kwargs["format"] == "NCL":
                return AclFormat.ACL_FORMAT_NCL
            return AclFormat.ACL_FORMAT_ND