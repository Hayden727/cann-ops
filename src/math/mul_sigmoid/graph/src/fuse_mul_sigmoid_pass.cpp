/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <iostream>
 #include "fp16_t.h"
 #include "register_custom_pass.h"
 #include "all_ops.h"
 
 using namespace std;
 using namespace ge;
 using ::op::fp16_t;
 
 namespace {
 const AscendString kOpTypeMul = "Mul";
 const AscendString kOpTypeSigmoid = "Sigmoid";
 const AscendString kOpTypeLessEqual = "LessEqual";
 const AscendString kOpTypeZerosLike = "ZerosLike";
 const AscendString kOpTypeSelect = "Select";
 const AscendString kOpTypeSub = "Sub";
 const AscendString kOpTypeStopGradient = "StopGradient";
 const AscendString kOpTypeAdd = "Add";
 const AscendString kOpTypeReshape = "Reshape";
 const AscendString kOpTypeConst = "Const";
 const AscendString kInputNameX2 = "x2";
 constexpr const char *kAttrNameT1 = "t1";
 constexpr const char *kAttrNameT2 = "t2";
 constexpr const char *kAttrNameT3 = "t3";
 constexpr int32_t MulIdx = 0;
 constexpr int32_t SigmoidIdx = 1;
 constexpr int32_t LessEqualIdx = 2;
 constexpr int32_t ZerosLikeIdx = 3;
 constexpr int32_t SelectIdx = 4;
 constexpr int32_t SubIdx = 5;
 constexpr int32_t StopGradientIdx = 6;
 constexpr int32_t AddIdx = 7;
 constexpr int32_t ReshapeIdx = 8;
 constexpr int32_t Mul1Idx = 9;
 constexpr int32_t Mul2Idx = 10;
 const vector<AscendString> nodeTypes = {kOpTypeMul, kOpTypeSigmoid, kOpTypeLessEqual, kOpTypeZerosLike, kOpTypeSelect, kOpTypeSub, kOpTypeStopGradient, kOpTypeAdd, kOpTypeReshape};
 const vector<tuple<int, AscendString, int, int>> edgeList = {
     make_tuple(MulIdx, kOpTypeSigmoid, SigmoidIdx, 1),
     make_tuple(SigmoidIdx, kOpTypeLessEqual, LessEqualIdx, 5),
     make_tuple(SigmoidIdx, kOpTypeZerosLike, ZerosLikeIdx, 5),
     make_tuple(SigmoidIdx, kOpTypeSelect, SelectIdx, 5),
     make_tuple(SigmoidIdx, kOpTypeSub, SubIdx, 5),
     make_tuple(SigmoidIdx, kOpTypeAdd, AddIdx, 5),
     make_tuple(LessEqualIdx, kOpTypeSelect, SelectIdx, 1),
     make_tuple(ZerosLikeIdx, kOpTypeSelect, SelectIdx, 1),
     make_tuple(SelectIdx, kOpTypeSub, SubIdx, 1),
     make_tuple(SubIdx, kOpTypeStopGradient, StopGradientIdx, 1),
     make_tuple(StopGradientIdx, kOpTypeAdd, AddIdx, 1),
     make_tuple(AddIdx, kOpTypeReshape, ReshapeIdx, 1),
     make_tuple(ReshapeIdx, kOpTypeMul, Mul1Idx, 1),
     make_tuple(Mul1Idx, kOpTypeMul, Mul2Idx, 1),
 };
 thread_local vector<GNode> nodeList(11);
 REG_OP(MulSigmoid)
     .INPUT(x1, TensorType({DT_FLOAT16}))
     .INPUT(x2, TensorType({DT_FLOAT16}))
     .REQUIRED_ATTR(t1, Float)
     .REQUIRED_ATTR(t2, Float)
     .REQUIRED_ATTR(t3, Float)
     .OUTPUT(out, TensorType({DT_FLOAT16}))
     .OP_END_FACTORY_REG(MulSigmoid)
 
 bool FindNodes(GraphPtr &graph) {
     size_t foundCount = 0;
     auto all_nodes = graph->GetAllNodes();
     for (auto &node: all_nodes) {
         AscendString node_type;
         auto ret = node.GetType(node_type);
         auto it = find(nodeTypes.begin(), nodeTypes.end(), node_type);
         if (it != nodeTypes.end()) {
             foundCount++;
             if (foundCount == nodeTypes.size()) {
                 break;
             }
         }
     }
     return foundCount == nodeTypes.size();
 }
 
 bool CheckOutNode(int nodeIdx, AscendString kOpType, int outNodeIdx, size_t nodeNum) {
     auto OutDataNodes = nodeList[nodeIdx].GetOutDataNodesAndPortIndexs(0);
     if (OutDataNodes.size() != nodeNum) {
         return false;
     }
     AscendString node_type;
     for (auto &[OutDataNode, _]: OutDataNodes) {
         auto ret = OutDataNode->GetType(node_type);
         if (node_type == kOpType) {
             nodeList[outNodeIdx] = *OutDataNode;
             return true;
         }
     }
     return false;
 }
 
 bool CheckNodesHaveEdge(GraphPtr &graph) {
     auto all_nodes = graph->GetAllNodes();
     for (auto &node: all_nodes) {
         AscendString node_type;
         auto ret = node.GetType(node_type);
         if (node_type != kOpTypeMul) {
             continue;
         }
         nodeList[MulIdx] = node;
         size_t checkedCount = 0;
         for (const auto& edge : edgeList) {
             if (apply(CheckOutNode, edge)) {
                 checkedCount++;
             } else {
                 break;
             }
         }
         if (checkedCount == edgeList.size()) {
             return true;
         }
     }
     return false;
 }
 
 fp16_t GetInputConstDataByName(int nodeIdx, const AscendString kInputName) {
     int32_t input_index;
     graphStatus ret = nodeList[nodeIdx].GetInputIndexByName(kInputName, input_index);
     Tensor const_data;
     ret = nodeList[nodeIdx].GetInputConstData(input_index, const_data);
     uint8_t* const_data_ptr = const_data.GetData();
     return static_cast<fp16_t>(*(reinterpret_cast<fp16_t*>(const_data_ptr)));;
 }
 
 void CreateMulSigmoidNode(GraphPtr &graph, GNode &node_mulsigmoid) {
     fp16_t t1 = GetInputConstDataByName(MulIdx, kInputNameX2);
     fp16_t t2 = GetInputConstDataByName(LessEqualIdx, kInputNameX2);
     fp16_t t3 = GetInputConstDataByName(Mul2Idx, kInputNameX2);
     AscendString kOpNameMulSigmoid;
     graphStatus ret = nodeList[Mul2Idx].GetName(kOpNameMulSigmoid);
     string kOpNameMulSigmoidString(kOpNameMulSigmoid.GetString());
     auto mulsigmoid = op::MulSigmoid((kOpNameMulSigmoidString + "_" + "mulSigmoid").c_str());
     mulsigmoid.SetAttr(kAttrNameT1, static_cast<float>(t1));
     mulsigmoid.SetAttr(kAttrNameT2, static_cast<float>(t2));
     mulsigmoid.SetAttr(kAttrNameT3, static_cast<float>(t3));
     node_mulsigmoid = graph->AddNodeByOp(mulsigmoid);
 }
 
 int32_t GetInDataNodeIdxByNotType(GNode &node, AscendString kOpType) {
     int32_t input_index = -1;
     for (size_t i = 0; i < node.GetInputsSize(); ++i) {
         auto [in_node, _] = node.GetInDataNodesAndPortIndexs(i);
         AscendString node_type;
         auto ret = in_node->GetType(node_type);
         if (node_type != kOpType) {
             input_index = i;
             break;
         }
     }
     return input_index;
 }
 
 bool AddInputsAndOutputs(GraphPtr &graph, GNode &node_mulsigmoid) {
     int32_t mul_input_index = GetInDataNodeIdxByNotType(nodeList[MulIdx], kOpTypeConst);
     if (mul_input_index == -1) {
         return false;
     }
     auto [mul_input_node, mul_output_index] = nodeList[MulIdx].GetInDataNodesAndPortIndexs(mul_input_index);
     auto ret = graph->AddDataEdge(*mul_input_node, mul_output_index, node_mulsigmoid, 0);
     TensorDesc input_desc_mul;
     ret = nodeList[MulIdx].GetInputDesc(mul_input_index, input_desc_mul);
     ret = node_mulsigmoid.UpdateInputDesc(0, input_desc_mul);
 
     int32_t mul1_input_index = GetInDataNodeIdxByNotType(nodeList[Mul1Idx], kOpTypeReshape);
     if (mul1_input_index == -1) {
         return false;
     }
     auto [mul1_input_node, mul1_output_index] = nodeList[Mul1Idx].GetInDataNodesAndPortIndexs(mul1_input_index);
     ret = graph->AddDataEdge(*mul1_input_node, mul1_output_index, node_mulsigmoid, 1);
     TensorDesc input_desc_mul1;
     ret = nodeList[Mul1Idx].GetInputDesc(mul1_input_index, input_desc_mul1);
     ret = node_mulsigmoid.UpdateInputDesc(1, input_desc_mul1);
 
     TensorDesc output_desc_mul2;
     ret = nodeList[Mul2Idx].GetOutputDesc(0, output_desc_mul2);
     ret = node_mulsigmoid.UpdateOutputDesc(0, output_desc_mul2);
     return true;
 }
 
 void RemoveOldNodesEdgesAndAddMulSigmoidOutput(GraphPtr &graph, GNode &node_mulsigmoid) {
     for (auto &node: nodeList) {
         for (size_t i = 0; i < node.GetInputsSize(); ++i) {
             auto [in_node, in_id] = node.GetInDataNodesAndPortIndexs(i);
             if (in_node != nullptr) {
                 auto ret = graph->RemoveEdge(*in_node, in_id, node, i);
             }
         }
     }
 
     for (auto &[out_node, out_id]: nodeList[Mul2Idx].GetOutDataNodesAndPortIndexs(0)) {
         if (out_node != nullptr) {
             auto ret = graph->RemoveEdge(nodeList[Mul2Idx], 0, *out_node, out_id);
             ret = graph->AddDataEdge(node_mulsigmoid, 0, *out_node, out_id);
         }
     }
 
     for (auto &node: nodeList) {
         auto ret = graph->RemoveNode(node);
     }
 }
 } // namespace
 
 /*
 before:
                Mul
                 |
              Sigmoid -  -  -
             /   |     \  \  \
            /    |      \  \  \
           /     |       \  \  \
    LessEqual ZerosLike  /  /   \
         \       |      /  /     \
          \      |     /  /      /
           \     |    /  /      /
            \    |   /  /      /
              Select   /      /
                 |    /      /
                 |   /      /
                 |  /      /
                Sub       /
                 |       /
           StopGradient /  
                 |     /  
                Add ——  
                 |
              Reshape
                 |
               Mul_1
                 |
               Mul_2
 
  after:
             MulSigmoid
 */
 graphStatus FuseMulSigmoidPass(GraphPtr &graph, CustomPassContext &custom_context) {
     cout << "FuseMulSigmoidPass begin." << endl;
     // 1.遍历所有节点，寻找nodeTypes中的节点
     if (!FindNodes(graph)) {
         cout << "Do not find every node in nodeTypes." << endl;
         return GRAPH_SUCCESS;
     }
 
     // 2.判断nodeTypes中的节点是否有连边关系
     if (!CheckNodesHaveEdge(graph)) {
         cout << "There is no edge like MulSigmoid." << endl;
         return GRAPH_SUCCESS;
     }
     
     do {
         // 3.创建和添加MulSigmoid节点
         GNode node_mulsigmoid;
         CreateMulSigmoidNode(graph, node_mulsigmoid);
 
         // 4.添加新节点的输入输出
         if (!AddInputsAndOutputs(graph, node_mulsigmoid)) {
             custom_context.SetErrorMessage("Add inputs and outputs failed.");
             return -1;
         }
 
         // 5.删除旧节点和其连边关系，连接新MulSigmoid节点和输出节点
         RemoveOldNodesEdgesAndAddMulSigmoidOutput(graph, node_mulsigmoid);
     } while ((FindNodes(graph)) && (CheckNodesHaveEdge(graph)));
     
     cout << "FuseMulSigmoidPass end." << endl;
     return GRAPH_SUCCESS;
 }
 
 REGISTER_CUSTOM_PASS("FuseMulSigmoidPass").CustomPassFn(FuseMulSigmoidPass);