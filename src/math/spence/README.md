## Spence
### 贡献说明
| 贡献者   | 贡献方  | 贡献算子     | 贡献时间     | 贡献内容       |
|-------|------|----------|----------|------------|
| 遗忘在冬天 | 社区任务 | Spence算子 | 2025/4/2 | 新增Spence算子 |

## 支持的产品型号
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

### 算子描述
- 原型信息

<table>  
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Spence</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td align="center">x</td><td align="center">4095</td><td align="center">float16, float32</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">4095</td><td align="center">float16, float32</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">spence</td></tr>  
</table>  

### 约束与限制
x,y的数据类型仅支持float16, float32，数据格式仅支持ND

### 算子使用
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

### 运行验证
跳转到对应调用方式目录，参考Readme进行算子运行验证。
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Spence算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/06/27 | 新增本readme |