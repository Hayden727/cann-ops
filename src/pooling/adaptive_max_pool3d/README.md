## `AdaptiveMaxPool3d`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`AdaptiveMaxPool3d`算子。

### 算子描述
`AdaptiveMaxPool3d`算子返回两个数据相加的结果。

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用AdaptiveMaxPool3d算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/AclOfflineModel"> AclOfflineModel</td><td>通过aclopExecuteV2调用的方式调用AdaptiveMaxPool3d算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/AclOnlineModel"> AclOnlineModel</td><td>通过aclopCompile调用的方式调用AdaptiveMaxPool3d算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/CppExtensions"> CppExtensions</td><td>Pybind方式调用AdaptiveMaxPool3d算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/PytorchInvocation"> PytorchInvocation</td><td>通过pytorch调用的方式调用AdaptiveMaxPool3d算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/TensorflowInvocation"> TensorflowInvocation</td><td>通过tensorflow调用的方式调用AdaptiveMaxPool3d算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/01/06 | 新增本readme |