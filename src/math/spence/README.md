## `Spence`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`Spence`算子。

### 算子规格描述

<table>  
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddCustom</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td align="center">x</td><td align="center">4095</td><td align="center">float16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">4095</td><td align="center">float16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>  
</table>  

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

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

### 算子调用
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用AddCustom算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/03 | 新增本readme |