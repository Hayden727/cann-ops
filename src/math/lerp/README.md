# Lerp
## 贡献说明
| 贡献者     | 贡献方  | 贡献算子 | 贡献时间      | 贡献内容     |
|---------|------|------|-----------|----------|
| enkilee | 社区任务 | Lerp | 2025/3/13 | 新增Lerp算子 |

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `Lerp`算子用对两个张量以`start`，`end`做线性插值，将结果返回到输出张量。

计算公式为：
  $$
  y=start+weight∗(end−start)
  $$

- 原型信息

  <table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Lerp</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="4" align="center">算子输入</td>
 
<tr><td align="center">start</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
<tr><td align="center">end</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
<tr><td align="center">weight</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">lerp</td></tr>  
  </table>

## 约束与限制
- start，end，weight，y，out的数据类型只支持float32,float16，数据格式只支持ND



## 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Lerp算子。</td>
    </tr>
</table>
