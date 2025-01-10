声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Addcdiv

## 支持的产品型号

Atlas A2 训练系列产品/Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：执行 x1 除以 x2 的元素除法，将结果乘以标量 value 并将其添加到 input_data 。

- 计算公式：

  $$
  out =input\_data+x1/x2×value
  $$

## 实现原理

调用`Ascend C`的`API`接口`Div`、`Mul`和`Add`进行实现。对于bfloat16的数据类型将其通过`Cast`和`ToFloat`接口转换为32位浮点数进行计算。


## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnAddcdivGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAddcdiv”接口执行计算。

* `aclnnStatus aclnnAddcdivGetWorkspaceSize(const aclTensor* inputData, const aclTensor* x1, const aclTensor* x2, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnAddcdiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnAddcdivGetWorkspaceSize

- **参数说明：**

  - inputData（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入input_data，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - tensor1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入tensor1，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - tensor2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入tensor2，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - value（aclScalar\*，计算输入）：必选参数，Host侧的aclScalar，公式中的输入value，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

 返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）：inputData、x1、x2、value、out的数据类型和数据格式不在支持的范围之内。
  ```

### aclnnAddcdiv

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddcdivGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- inputData、x1、x2和out的数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式只支持ND

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Addcdiv</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="5" align="center">算子输入</td>
<tr><td align="center">intput_data</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td align="center">x1</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td align="center">x2</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td align="center">value</td><td align="center">scalar</td><td align="center">float32,float16,bfloat16</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">addcdiv</td></tr>  
</table>

## 调用示例

详见[Addcdiv自定义算子样例说明算子调用章节](../README.md#算子调用)
