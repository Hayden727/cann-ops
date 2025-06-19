## `ScatterReduce`自定义算子样例说明

本样例通过`Ascend C`编程语言实现了`ScatterReduce`算子。

算子描述

`ScatterReduce`算子将源张量（`src`）的值按照索引（`index`）规则归约到目标张量（`x`）的指定维度（`dim`），支持多种归约操作。

算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ScatterReduce</th></tr>  
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="3" align="center">算子输入</td>
    <td align="center">x</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>  
    <td align="center">index</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr>  
    <td align="center">src</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>  
<tr><td rowspan="3" align="center">attr属性</td>
    <td align="center">dim</td><td align="center">int</td><td colspan="2" align="center">required</td></tr>
    <td align="center">reduce</td><td align="center">str</td><td colspan="2" align="center">required</td></tr>
    <td align="center">include_self</td><td align="center">bool</td><td colspan="2" align="center">default: TRUE</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">scatter_reduce</td></tr>  
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
```

### 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 算子包编译部署
  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/cann-ops
    ```

  - 执行编译

    ```bash
    bash build.sh
    ```

  - 部署算子包

    ```bash
    bash build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
    ```
### 算子调用
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用ScatterReduce算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/01/08 | 新增本readme |