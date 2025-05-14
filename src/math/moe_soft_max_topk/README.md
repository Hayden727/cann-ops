## `MoeSoftMaxTopk`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`MoeSoftMaxTopk`算子。

MoeSoftMaxTopk是softmax和topk的融合算子，其中softmax可以理解为对x计算最后一维每个数据的概率，在计算结果中筛选出k个最大结果，输出对应的y值和索引indices。  
计算公式如下：
$$ softmax(x_{i} )=\frac{exp(x_{i} )}{\sum exp(x_{j} )} $$
topk是对sofrmax的所有结果进行一维选择，获取最大的k个结果，并输出对应的值y和索引indices。

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">MoeSoftMaxTopk</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>
<tr><td align="center">x</td><td align="center">1024 * 16</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>

</tr>
</tr>
<tr><td rowspan="3" align="center">算子输出</td>

<tr><td align="center">y</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">indices</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
</tr>
<tr><td rowspan="1" align="center">attr属性</td><td align="center">k</td><td align="center">\</td><td align="center">int</td><td align="center">\</td><td align="center">4</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">moe_soft_max_topk</td></tr>
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用MoeSoftMaxTopk算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/05/08 | 新增本readme |