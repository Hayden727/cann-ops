## 目录结构介绍
``` 
├── AclNNInvocationNaive              //通过aclnn调用的方式调用Sqrt算子
│   ├── gen_data.py                   // 输入数据和真值数据生成脚本
│   ├── verify_result.py              // 真值对比文件
│   ├── main.cpp                      // 单算子调用应用的入口
│   └── run.sh                        // 执行命令脚本
``` 
## 代码实现介绍
完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。src/main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。    

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：
   ```cpp    
   aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor *x, const alcTensor *y, uint64_t workspaceSize, aclOpExecutor **executor);
   aclnnStatus aclnnSqrt(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);
   ```
其中aclnnSqrtGetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnSqrt执行计算。具体参考[AscendCL单算子调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)>单算子API执行 章节。

## 运行样例算子
### 1.&nbsp;编译算子工程
运行此样例前，请完成算子包编译部署，请参考[算子包编译部署](../../README.md#算子包编译部署)。

### 2.&nbsp;aclnn调用样例运行

  - 进入到样例目录

    ```bash
    cd ${git_clone_path}/src/math/sqrt/examples/AclNNInvocation
    ```
  - 样例执行    

    样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后检验运行结果。具体过程可参见run.sh脚本。

    ```bash
    bash run.sh
    ```

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2024/12/30 | 新增本readme |
| 2025/01/06 | 更新本readme |