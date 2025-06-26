# `Neg`自定义算子样例说明  
本样例通过`Ascend C`编程语言实现了`Neg`算子。

### 算子描述
`Neg`算子对输入的数值型数据执行取负操作（y = -x）。

### 算子规格描述

<table> 
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Neg</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr> <tr><td rowspan="1" align="center">算子输入</td> 
<td align="center">x</td><td align="center">tensor</td> <td align="center">int32, int8, float16, bfloat16, float32</td><td align="center">ND</td></tr> 

<tr><td rowspan="1" align="center">算子输出</td> 
<td align="center">y</td><td align="center">tensor</td> <td align="center">与输入相同</td><td align="center">ND</td></tr> 
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">neg</td></tr> 
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 推理系列产品

### 目录结构介绍
```
├── docs                     // 算子文档目录
├── example                  // 调用示例目录
├── framework                // 第三方框架适配目录
├── op_host                  // host目录
├── op_kernel                // kernel目录
├── opp_kernel_aicpu         // aicpu目录
└── tests                    // 测试用例目录
```

### 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 算子包编译部署
1. 进入到仓库目录
   ```bash
   cd ${git_clone_path}/cann-ops
   ```

2. 执行编译
   ```bash
   bash build.sh
   ```

3. 部署算子包
   ```bash
   bash build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
   ```

### 算子调用
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Neg算子</td>
    </tr>
</table>

## 更新说明
| 时间       | 更新事项 |
|------------|----------|
| 2025/06/23 | 新增Neg算子说明文档 |