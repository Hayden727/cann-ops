## `MatmulLeakyReluCustom`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`MatmulLeakyReluCustom`算子。

### 算子描述
`MatmulLeakyReluCustom`算子使用了MatmulLeakyRelu高阶API，实现了快速的MatmulLeakyRelu矩阵乘法的运算操作。

MatmulLeakyReluCustom的计算公式为：

```
C = A * B + Bias
C = C > 0 ? C : C * 0.001
```

- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。


### 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">MatmulLeakyReluCustom</th></tr>

<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="4" align="center">算子输入</td>
 
<tr>
<td align="center">a</td><td align="center">tensor</td><td align="center">float16</td><td align="center">ND</td></tr>

<tr>
<td align="center">b</td><td align="center">tensor</td><td align="center">float16</td><td align="center">ND</td></tr>

<tr>
<td align="center">bias</td><td align="center">tensor</td><td align="center">float32</td><td align="center">ND</td>
</tr>  

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">c</td><td align="center">tensor</td><td align="center">float32</td><td align="center">ND</td></tr>


<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_leakyrelu_custom</td></tr>  
</table>


### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品


### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用MatmulLeakyRelu算子。</td>
    </tr>
</table>

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/06/20 | 新增本readme |