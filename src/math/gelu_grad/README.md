# GeluGrad
### 贡献说明
| 贡献者   | 贡献方              | 贡献算子     | 贡献时间      | 贡献内容         |
|-------|------------------|----------|-----------|--------------|
| zhang | 西北工业大学-智能感知交互实验室 | GeluGrad | 2025/1/10 | 新增GeluGrad算子 |

### 支持的产品型号
- Atlas A2训练系列产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

### 算子描述

- 功能描述

`GeluGrad`算子用于计算Gelu函数的梯度。
- 计算公式：

  - **Gelu函数**

    $$
    y=\frac{x}{exp((c_{0}x^{2}+c_{1})x)+1}
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759$
  - **对于Atlas A2 训练系列产品:**

    $$
    px=exp((x^{2}\times c_{0}+c_{1})\times x)
    $$
    $$
    res0=(x^{2}\times c_{2}+c_{3})\times x
    $$
    $$
    t=\frac{1}{px+1}
    $$
    $$
    z=(px\times res0\times t^{2}+t)\times dy
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759,c_{2}=0.2140644488178007,c_{3}=1.595769121605730711759$
  - **对于Atlas 200I/500 A2推理产品：**

    $$
    g1=\frac{1}{exp((x^{2}\times c_{0}+c_{1})\times x)+1}
    $$
    $$
    g2=x^{2}\times c_{2}+c_{3}
    $$
    $$
    z=((((g1-1)\times x)\times g2+1)\times g1)\times dy
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.5957691216057308,c_{2}=-0.21406444881780074632901625683959062,c_{3}=-1.5957691216057307117597842397375274738$

- 原型信息

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GeluGrad</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">dy</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td>
</tr> 
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td>
</tr> 
<tr><td rowspan="1" align="center">算子输入</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">z</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>    
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gelu_grad</td></tr>  
</table>

### 约束与限制
- dy，x,y,z,out的数据类型仅支持float32,float16,bfloat16，数据格式仅支持ND

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

### 算子使用
使用该算子前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。


### 编译部署
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

### 运行验证
跳转到对应调用方式目录，参考Readme进行算子运行验证。
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用GeluGrad算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/06/27| 新增本readme |