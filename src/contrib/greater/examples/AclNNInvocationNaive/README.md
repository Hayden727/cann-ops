## 概述
通过aclnn调用的方式调用Greater算子。

## 目录结构介绍
``` 
├── AclNNInvocationNaive
│   ├── CMakeLists.txt      // 编译规则文件
│   ├── gen_data.py         // 算子期望数据生成脚本
│   ├── main.cpp            // 单算子调用应用的入口
│   ├── run.sh              // 编译运行算子的脚本
│   └── verify_result.py    // 计算结果精度比对脚本
``` 
## 代码实现介绍
完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。    

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：
   ```cpp    
   aclnnStatus aclnnGreaterGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);
   aclnnStatus aclnnGreater(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);
   ```

## 运行样例算子
  **请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/cann-ops/src/contrib/greater/examples/AclNNInvocationNaive
    ```

  - 环境变量配置
    
    需要设置环境变量，以arm为例
    
    ```bash
    export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib
    ```
  - 样例执行
    
    样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后打印运行结果。
    
    ```bash
    mkdir -p build
    cd build
    cmake .. && make
    ./execute_greater_op
    ```
    
    用户亦可参考run.sh脚本进行编译与运行。

    ```bash
    bash run.sh
    ```