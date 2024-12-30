## `Sqrt`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`Sqrt`算子。

## 算子描述
`Sqrt`算子将数据进行开方运算。

## 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Sqrt</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
 
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sqrt</td></tr>  
</table>

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
| 2024/12/24 | 新增本readme |
| 2024/12/30 | 更新本readme |