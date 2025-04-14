声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# GeluQuant

## 支持的产品型号

Atlas 训练系列产品/Atlas 推理系列产品/Atlas A2训练系列产品/Atlas 800I A2推理产品/Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 实现原理

GeluQuant由Gelu算子和Quant量化操作组成，计算过程：

1. gelu = Gelu(x, approximate)
2. quant_mode 是static, y = Round(gelu*scale+offset).clip(-128, 127)
3. quant_mode 是dynamic, y=  (gelu*scale) * (127.0 / max(abs(gelu*scale)));
out_scale = max(abs(gelu*scale))/127.0

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnGeluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGeluQuant”接口执行计算。

* `aclnnStatus aclnnGeluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *inputScaleOptional, const aclTensor *inputOffsetOptional, char *approximateOptional, char *quantModeOptional, const aclTensor *yOut, const aclTensor *outScaleOut, uint64_t *workspaceSize,
aclOpExecutor **executor)`
* `aclnnStatus aclnnGeluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnGeluQuantGetWorkspaceSize

- **参数说明：**
  
  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16，FLOAT32, BF16 [数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND, 输入shape至少2维，至多8维。
  - inputScaleOptional（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，公式中的输入scale，数据类型支持FLOAT16，FLOAT32, BF16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。当quantMode 是static, 则是必选参数。shape只可以是一维，大小可以是x的shape的最后一个维度，或者1。
  - inputOffsetOptional（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，公式中的输入offset，数据类型支持FLOAT16，FLOAT32, BF16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。shape和数据类型应该和inputScaleOptional一致。
  - approximateOptional（char\*，计算输入）：可选参数，公式中的输入approximate，数据类型支持STRING，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)。值必须是tanh 或者none。
  - quantModeOptional（char\*，计算输入）：可选参数，公式中的输入quant_mode，数据类型支持STRING，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)。值必须是dynamic 或者static。
  - yOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持INT8，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。Shape 和输入x一致。
  - outScaleOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出out_scale，数据类型支持FLOAT32，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。Shape 和输入x的shape除了最后一个维度，其他维度都一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、inputScaleOptional、inputOffsetOptional的数据类型和数据格式不在支持的范围内。
  ```

### aclnnGeluQuant

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddCustomGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- x、inputScaleOptional、inputOffsetOptional的数据类型只支持FLOAT16，FLOAT32, BF16，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">GeluQuant</td></tr>
</tr>
<tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">(M,K1)</td><td align="center">float16，float32, bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">scale</td><td align="center">(K1)</td><td align="center">float16，float32, bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">offset</td><td align="center">(K1)</td><td align="center">float16，float32, bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">approximate</td><td align="center">(1)</td><td align="center">string</td><td align="center">ND</td></tr>
<tr><td align="center">quant_mode</td><td align="center">(1)</td><td align="center">string</td><td align="center">ND</td></tr>

</tr>
</tr>
<tr><td rowspan="2" align="center">算子输出</td><td align="center">y</td><td align="center">(M,K1)</td><td align="center">int8</td><td align="center">ND</td></tr>
<tr><td align="center">out_scale</td><td align="center">(M)</td><td align="center">float32</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gelu_quant</td></tr>
</table>

## 调用示例

详见[GeluQuant自定义算子样例说明算子调用章节](../README.md#算子调用)
