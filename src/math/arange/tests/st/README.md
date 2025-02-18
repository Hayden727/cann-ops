## 目录结构介绍
```
├── msopst.ini                 // st测试配置文件 
├── Arange_case_alltype.json   // 测试用例定义文件示例(8.0.RC3.alpha003版本生成)
└── test_arange.py             // 算子期望数据生成脚本
```

## ST测试介绍

完成算子包部署后，可选择使用msOpST工具进行ST（System Test）测试，在真实的硬件环境中，对算子的输入输出进行测试，以验证算子的功能是否正确。

测试用例通常包括各种不同类型的数据输入和预期输出，以及一些边界情况和异常情况的测试。通过ST测试，可以确保算子功能的正确性，并且能够在实际应用中正常运行。

具体描述可参考[算子测试（msOpST）
](https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/ODtools/Operatordevelopmenttools/msopdev_16_0087.html)章节。

## 执行测试用例
  **请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**

  - 配置环境变量

    ```bash
    export DDK_PATH=${INSTALL_DIR}
    export NPU_HOST_LIB=${INSTALL_DIR}/{arch-os}/lib64
    ```

  - 进入到测试用例目录

    ```bash
    cd ${git_clone_path}/ops-contribution/src/math/arange/tests/st
    ```

  - 根据执行机器的架构修改msopst.ini中的atc_singleop_advance_option和HOST_ARCH

  - 查看Soc Version

    ```bash
    npu-smi info
    ```
    打印的表格中Name列即为Soc Version

  - 执行测试用例

    ```bash
    ${INSTALL_DIR}/python/site-packages/bin/msopst run -i ./Arange_case_alltype.json -soc {Soc Version} -out ./output -conf msopst.ini
    ```
## 注意事项
- st测试bfloat16数据类型时提示fail，是由于当前环境下tensorflow bfloat16类型arange算子输出不对，本实现的arange算子输出是正确的。
- arange算子的输出shape依赖实际的计算，当前没有好的办法，暂时，host侧的infershape中SetDim需要根据测试样例数值修改。

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/02/18 | 新增本readme |
