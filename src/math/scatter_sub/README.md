## `ScatterSub`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`SatterSub`算子。

### 算子描述
`ScatterSub`算子将`var`中的数据用`indices`进行索引，索引结果与`updates`进行减法操作。具体计算方式如下：

```
# Scalar indices
var[indices, ...] -= updates[...]

# Vector indices (for each i)
var[indices[i], ...] -= updates[i, ...]

# High rank indices (for each i, ..., j)
var[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
```

### 算子规格描述

<table>  
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">ScatterSub</th></tr>  
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
<tr><td align="center">var</td><td align="center">-</td><td align="center">float3,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">indices</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">updates</td><td align="center">-</td><td align="center">float3,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td><td align="center">var</td><td align="center">-</td><td align="center">float3,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">attr属性</td><td align="center">use_locking</td><td align="center">\</td><td align="center">bool</td><td align="center">\</td><td align="center">false</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="8" align="center">scattersub</td></tr>  
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用ScatterSub算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/05/16 | 新增本readme |
