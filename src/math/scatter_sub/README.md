# ScatterSub
## 贡献说明

| 贡献者 | 贡献方  | 贡献算子       | 贡献时间      | 贡献内容           |
|-----|------|------------|-----------|----------------|
| JY  | 社区任务 | ScatterSub | 2025/5/16 | 新增ScatterSub算子 |

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `ScatterSub`算子将`var`中的数据用`indices`进行索引，索引结果与`updates`进行减法操作。具体计算方式如下：

 ```
# Scalar indices
var[indices, ...] -= updates[...]

# Vector indices (for each i)
var[indices[i], ...] -= updates[i, ...]

# High rank indices (for each i, ..., j)
var[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
```

- 原型信息

  <table>  
    <tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">ScatterSub</th></tr>  
    <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
    <tr><td align="center">var</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
    <tr><td align="center">indices</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td><td align="center">\</td></tr>  
    <tr><td align="center">updates</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">var</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>
    <tr><td align="center">attr属性</td><td align="center">use_locking</td><td align="center">\</td><td align="center">bool</td><td align="center">\</td><td align="center">false</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="8" align="center">scatter_sub</td></tr>  
  </table>

## 约束与限制
- var，indices，uodates，out的数据类型只支持float32,float16,int32,int8，数据格式只支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用ScatterSub算子。</td>
    </tr>
</table>
