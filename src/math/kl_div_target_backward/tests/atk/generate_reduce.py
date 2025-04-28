
import random


from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("reduce")  # reduce为注册的生成器名称，对应yaml中的generate参数
class ReduceGenerator(CaseGenerator):

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        '''
        用例参数约束修改入口
        :param case_config:  生成的用例信息，可能不满足参数间约束，导致用例无效
        :return: 返回修改后符合参数间约束关系的用例，需要用例保障用例有效
        '''
        dim = len(case_config.inputs[0].shape)  # 获取第一个tensor参数shape最大dim值
        range_is_null = case_config.inputs[0].is_range_null()  # 判断是否为空tensor
        dtype=case_config.inputs[0].dtype
        case_config.inputs[1].dtype=dtype
        case_config.inputs[2].dtype=dtype
        shape=case_config.inputs[0].shape
        case_config.inputs[1].shape=shape
        case_config.inputs[2].shape=shape
        return case_config  # 返回修改和符合参数约束的用例
