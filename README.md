![输入图片说明](https://foruda.gitee.com/images/1732709982038009684/f1bee069_9519913.jpeg "首页banner.jpg")

## 🎯 项目介绍
ops-contribution是昇腾与生态伙伴共建的开放仓库，欢迎开发者体验基于昇腾平台提供的系列算子代码样例。

## 🔍 仓库结构
ops-contribution仓关键目录如下所示：
```
├── cmake
├── src // 算子源码目录
│ ├── common // 公共目录
│ ├── math // 数学库算子目录
│ │ ├── add_custom // AddCustom算子目录
│ └── CMakeLists.txt
├── CMakeLists.txt
├── CMakePresets.json // 配置文件
├── README.md
└── build.sh // 算子编译脚本
```
## ⚡️ 快速上手
<table>
<tr><td width="15%"><b>样例名称</b></td><td width="15%"><b>样例介绍</b></td><td width="40%"><b>开发语言</b></td></tr>
<tr><td><a>add_custom</a></td><td>参考算子，基于Ascend C的vector加法算子</td><td>Ascend C</td></tr>
</table>

## 📝 版本配套说明
- 请参考[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0001.html)相关章节，对昇腾硬件、CANN软件及相应深度学习框架进行安装准备。

## 📌 社区交流
了解更多资源，欢迎访问[昇腾社区Ascend C](https://www.hiascend.com/ascend-c)

#### **📖 学习教程**
- **👉 [Ascend C算子开发（入门）](https://www.hiascend.com/developer/courses/detail/1691696509765107713)**
- **👉 [Ascend C算子开发（进阶）](https://www.hiascend.com/developer/courses/detail/1696414606799486977)**

#### **🔥 系列直播 | 深度开放特辑**
- **直播平台**：【昇腾CANN】视频号、[B站【昇腾AI开发者】](https://space.bilibili.com/1190614918?spm_id_from=333.337.search-card.all.click)、[昇腾社区同步直播](https://www.hiascend.com/developer/cann20242?tab=live)<br>
- **回放地址**：https://www.bilibili.com/video/BV1ouBMYpEJC/?spm_id_from=333.999.0.0 <br>
- **直播预告**：<br>
![输入图片说明](resouce/%E7%9B%B4%E6%92%AD%E9%A2%84%E5%91%8A-2%E6%9C%88.png)

#### **🏅️ 项目发放**
`传送门`：**项目发放列表**<br>
 _*项目发放持续更新中，敬请期待_ 

#### **💌 联系我们**
若您对开放仓库的使用有任何建议和疑问，欢迎发送邮件到cann@huawei.com。<br>
 :globe_with_meridians: 网站：https://www.hiascend.com/software/cann <br>
 :mailbox_with_mail: 邮箱：cann@huawei.com <br>
 :speech_balloon: 论坛：https://www.hiascend.com/forum/forum-0106101385921175004-1.html <br>

## 🤝 共建伙伴
[![输入图片说明](resouce/%E5%85%B1%E5%BB%BA%E4%BC%99%E4%BC%B4-4%E4%B8%AA.png)](http://https://gitee.com/Nicet)

## ⭐️ 贡献者
我们非常欢迎您为CANN贡献代码，也非常感谢您的反馈。<br>
[![输入图片说明](resouce/%E5%BC%A0%E5%BF%97%E4%BC%9F-CIRCLE.png)](https://gitee.com/Nicet)

- 贡献算子列表
<table>
<tr><td width="15%"><b>算子分类</b></td><td width="15%"><b>算子</b></td><td width="40%"><b>简介</b></td><td width="30%"><b>贡献者</b></td></tr>
<tr><td>Sqrt</td><td>Sqrt</td><td>Sqrt原生自定义算子，实现了对输入数据计算开方，获取输出数据的功能。</td><td> <a href="https://gitee.com/Nicet">Nice_try</a><br>西北工业大学-智能感知交互实验室</td></tr>
</table>

## 📝 贡献指南
ops-contribution仓欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见<a> **Contributing** </a>了解行为准则，进行CLA协议签署，以及参与源码仓贡献的详细流程。<br>

**针对ops-contribution仓，开发者准备本地代码与提交PR时需要重点关注如下几点**：<br>

1.提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。<br>
2.若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。
