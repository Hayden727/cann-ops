## 目录结构介绍
```
├── msopst.ini                  // st测试配置文件 
├── Addcmul_case_alltype.json     // 测试用例定义文件示例(8.0.RC3.alpha003版本生成)
└── test_addcmul.py               // 算子期望数据生成脚本
```

## ST测试介绍

完成算子包部署后，可选择使用msOpST工具进行ST（System Test）测试，在真实的硬件环境中，对算子的输入输出进行测试，以验证算子的功能是否正确。

测试用例通常包括各种不同类型的数据输入和预期输出，以及一些边界情况和异常情况的测试。通过ST测试，可以确保算子功能的正确性，并且能够在实际应用中正常运行。

具体描述可参考[算子测试（msOpST）
](https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/ODtools/Operatordevelopmenttools/msopdev_16_0087.html)章节。

## 执行测试用例
### 1.&nbsp;编译算子工程
运行测试用例前，请完成算子包编译部署。
  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}
    ```

  - 执行编译

    ```bash
    ./build.sh --disable-check-compatible
    ```

  - 部署算子包

    ```bash
    ./build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
    ```
### 2.&nbsp;生成测试用例

  - 进入到测试用例目录

    ```bash
    cd ${git_clone_path}/src/math/addcmul/tests/st
    ```

  - 生成测试用例

    ```bash
    ${INSTALL_DIR}/python/site-packages/bin/msopst create -i ../../op_host/addcmul.cpp -out ./
    ```
### 3.&nbsp;执行测试用例
  - 配置环境变量

    ```bash
    export DDK_PATH=${INSTALL_DIR}
    export NPU_HOST_LIB=${INSTALL_DIR}/{arch-os}/devlib
    ```

  - 进入到测试用例目录

    ```bash
    cd ${git_clone_path}/src/math/addcmul/tests/st
    ```

  - 执行测试用例

    ```bash
    ${INSTALL_DIR}/python/site-packages/bin/msopst run -i ./Addcmul_case_timestamp.json -soc {Soc Version} -out ./output
    ```

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2024/12/30 | 新增本readme |
| 2024/12/31 | 更新本readme |
