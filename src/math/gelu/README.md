# Gelu
### 贡献说明
| 贡献者   | 贡献方  | 贡献算子 | 贡献时间      | 贡献内容     |
|-------|------|------|-----------|----------|
| Mrkey | 神州鲲泰 | Gelu | 2025/3/17 | 新增Gelu算子 |

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 200/500 A2 推理产品
- Atlas A2训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述

- 功能描述

Gelu（Gaussian Error Linear Unit）是神经网络中常用的激活函数。Gelu是基于高斯误差函数定义的，相较于ReLU等激活函数，Gelu更加平滑，有助于提高训练过程的收敛速度和性能。

$$Gelu(x) = x\times \Phi(x)$$

$$\Phi(x) = \frac{1}{2} \times (1+elf(\frac{x}{\sqrt{2} }))$$
其中$elf(x)$为高斯误差函数。

但是高斯误差函数无法直接计算，学者们提出了一种近似计算高斯误差函数的方法，即：

$$\text{GELU}(x) \approx \frac{x}{1 + \exp\left(-\sqrt{\frac{8}{\pi}} \left(x + 0.044715 \cdot x^3\right)\right)}$$

- 原型信息

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">Gelu</th></tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>

<tr><td align="center">x</td><td align="center">-</td><td align="center">float32, float16, bfloat16</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">-</td><td align="center">float32, float16, bfloat16</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">gelu</td></td></tr>
</table>

### 约束与限制
x,y，out的数据类型仅支持float32, float16, bfloat16，输出仅支持ND

### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
└── opp_kernel_aicpu            // aicpu目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Gelu算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/06/27 | 新增本readme |