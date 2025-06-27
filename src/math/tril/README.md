## Tril 
### 贡献说明
| 贡献者     | 贡献方  | 贡献算子 | 贡献时间      | 贡献内容     |
|---------|------|------|-----------|----------|
| enkilee | 社区任务 | Tril | 2025/3/11 | 新增Tril算子 |

### 支持的产品型号
- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品
产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)


### 算子描述
- 功能描述
`Tril`算子用于提取张量的下三角部分。返回一个张量`out`，包含输入矩阵(2D张量)的下三角部分，`out`其余部分被设为0。这里所说的下三角部分为矩阵指定对角线`diagonal`之上的元素。参数`diagonal`控制对角线：默认值是`0`，表示主对角线。如果 `diagonal > 0`，表示主对角线之上的对角线；如果 `diagonal < 0`，表示主对角线之下的对角线。

计算公式为：
  $$
  y = tril(x, diagonal)
  $$
- 原型信息

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Tril</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
 
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">diagonal</td><td align="center">diagonal</td><td align="center">int</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">tril</td></tr>  
</table>

### 约束与限制
- name,tensor,x,y,out的数据类型仅支持float32,float16，数据格式只支持ND


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


### 算子使用
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


### 运行验证
跳转到对应调用方式目录，参考Readme进行算子运行验证。
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Tril算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/ATBInvocation"> ATBInvocation</td><td>通过ATB调用的方式调用Tril算子。</td>
    </tr>

</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/06/25| 新增本readme |
