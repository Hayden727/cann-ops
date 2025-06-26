# Cast 

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|----|----|----|------|------|
| 郭士鼎 | CANN生态 | Cast | 2025/6/25 | 新增Cast算子。|

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 算子描述

- 功能描述    

  Cast算子提供将tensor从源数据类型转换为目标数据类型的功能。

- 原型信息    

    <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cast</td></tr>
    <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">x</td><td align="center">-</td><td align="center">float16, float32, int32, int8, uint8, bool, int64, bfloat16, int16</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">-</td><td align="center">float16, float32, int32, int8, uint8, bool, int64, bfloat16, int16</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">算子属性</td><td align="center">dstType</td><td align="center">-</td><td align="center">int64</td><td align="center">\</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cast</td></td></tr>
    </table>

## 约束与限制

- dstType需要与out的数据类型一致，具体可以参考(https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/basicdataapi/atlasopapi_07_00514.html)
- x，out的数据格式只支持ND
- x，out具体支持的数据类型如下表所示

    <table>
    <tr>
    <td></td><td></td><td colspan="9" align="center">目标数据类型(out)</td>
    </tr>
    <tr>
    <td></td><td></td><td align="center">float16</td><td align="center">float32</td><td align="center">int32</td><td align="center">int8</td><td align="center">uint8</td><td align="center">bool</td><td align="center">int64</td><td align="center">bfloat16</td><td align="center">int16</td>
    </tr>
    <tr>
    <td rowspan="9" align="center">源数据类型(x)</td><td align="center">float16</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">float32</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">int32</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">int8</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">uint8</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">bool</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center"></td>
    </tr>
    <tr>
    <td align="center">int64</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">bfloat16</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center"></td><td align="center"></td>
    </tr>
    <tr>
    <td align="center">int16</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center"></td><td align="center"></td>
    </tr>
    </table>

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Cast算子。</td>
    </tr>
</table>