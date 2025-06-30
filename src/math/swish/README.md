# Swish
## 贡献说明
| 贡献者    | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|--------|------------------|-------|-----------|-----------|
| Yxymay | 西北工业大学-智能感知交互实验室 | Swish | 2025/1/14 | 新增Swish算子 |

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

`Swish`算子实现Swish激活函数，是一种由输入与其经过Sigmoid函数结果相乘得到的平滑、非线性函数，计算公式为：
$$y=x\cdot\mathrm{sigmoid}\left(s\cdot x\right)=x\cdot\frac{1}{1+e^{-s\cdot x}}$$
该激活函数具有良好的梯度传播特性，有助于提高深度神经网络的训练性能。

- 原型信息

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Swish</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
 
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">scale</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">swish</td></tr>  
</table>

## 约束与限制
- x,y的数据类型仅支持float32,float16,bfloat1，数据格式仅支持ND

## 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
```

## 算子使用
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Swish算子。</td>
    </tr>
</table>
