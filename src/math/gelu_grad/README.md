## `GeluGrad`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`GeluGrad`算子。

## 算子描述
`GeluGrad`算子用于计算`Gelu`函数的梯度，`Gelu`的具体公式为$y=\frac{x}{exp((c_{0}x^{2}+c_{1})x)+1}(c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759)$。

## 算子规格描述

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

## 计算公式
#### 1.对于AscendC910B设备
$$px=exp((x^{2}\times c_{0}+c_{1})\times x)$$
$$res0=(x^{2}\times c_{2}+c_{3})\times x$$
$$t=\frac{1}{px+1}$$
$$resp=(px\times res0\times t^{2}+t)\times dy$$
其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759,c_{2}=0.2140644488178007,c_{3}=1.595769121605730711759$
#### 2.对于AscendC310B设备
$$g1=\frac{1}{exp((x^{2}\times c_{0}+c_{1})\times x)+1}$$
$$g2=x^{2}\times c_{2}+c_{3}$$
$$res=((((g1-1)\times x)\times g2+1)\times g1)\times dy$$
其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.5957691216057308,c_{2}=-0.21406444881780074632901625683959062,c_{3}=-1.5957691216057307117597842397375274738$

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品
- Atlas 200/500 A2推理产品

## 目录结构介绍
```
├── examples    // 调用示例
├── op_host    // host侧实现文件
└── op_kernel  // kernel侧实现文件
```

## 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2024/12/30 | 新增本readme |