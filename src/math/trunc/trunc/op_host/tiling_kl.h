/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifndef TILING_KUNLUN_H
#define TILING_KUNLUN_H

#include <memory>
#include <cmath>

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "base_kl.h"
#include "tiling_algorithm.h"


namespace kunlun{
namespace tiling{
    using ExSpaceConsumer = std::function<Int(Int)>;

    // 所在位置
    enum class Position : std::uint8_t {
        INPUT = 0,   // 输入位置
        OUTPUT = 1,  // 输出位置
        CALC = 2     // 计算位置
    };

    // NPU 的 核心细节
    struct CoreDetail {
        Int coreNum;           // 核心数量
        Int memSize;            // 统一缓冲区大小
    };

    // 队列 信息
    struct Queue {
        Position pos;               // 队列位置
        Int lengthWeight;           // 长度权重（如：2个输入每次搬入长度分别为2000：1000时，可以分别将比重配置为2：1）
        Int dtSize;                 // 数据类型大小（如：bool/int8->1, half->2, int32/float->4）
        Int totalLength;            // 总长度（一般仅IO配置）
        Int formercoreLen;          // 大核 核内处理长度（一般仅IO配置）
        Int fc_formerTileLen;       // 大核 大Tile长度
        Int fc_tailTileLen;         // 大核 小Tile长度
        Int tailcoreLen;            // 小核 核内处理长度（一般仅IO配置）
        Int tc_formerTileLen;       // 小核 大Tile长度
        Int tc_tailTileLen;         // 小核 小Tile长度
    };
    
    // 队列池 (核内)
    class QueuePool {
    public:
        // 设置可用空间
        void setTotalSize(Int newTotalSize) {
            this->totalSize = newTotalSize;
        }

        // 添加队列
        void add(const Queue& que) {
            pool.emplace_back(que);
            if (que.dtSize < alignSize)
                alignSize = que.dtSize;
        }

        // 清空队列池
        void clear() {
            pool.clear();
        }

        // 注册额外空间分配器
        void registerExtraSpaceConsumer(ExSpaceConsumer consumer) { 
            exConsumerList.emplace_back(consumer); 
        }

        // 开/关 DoubleBuffer
        void setDoubleBufferEnable(const bool& newDoubleBufferEnabled) {
            this->doubleBufferEnabled = newDoubleBufferEnabled;
        }

        // 获取Queue总比重 (数据类型大小*长度比重*DoubleBuffer)
        Int totalWeight() const {
            Int ret = 0;
            if (this->doubleBufferEnabled) {
                for (auto que : pool) {
                    ret += que.dtSize * que.lengthWeight * (que.pos == Position::CALC ? 1 : 2);
                }
            } else {
                for (auto que : pool) {
                    ret += que.dtSize * que.lengthWeight;
                }
            }
            return ret;
        }

        // 将空间分区 (向下N对齐)
        template<Int N>
        Int partionLength_AF_N() const {
            const Int weight = this->totalWeight();
            const Int nLength = N / alignSize;
            if(weight == 0 || nLength == 0){
                return 0;
            }
            Int partionLength_AF_N = totalSize / weight / nLength * nLength;
            Int remainSize;

            AVOID_MEM_OVERFLOW: {
                remainSize = this->totalSize; 
                for (auto& consumer : exConsumerList) {
                    auto extraConsumed_AC32 = (consumer(partionLength_AF_N) + 31) / 32 * 32;
                    remainSize -= extraConsumed_AC32;
                }
                if (partionLength_AF_N * weight > remainSize) {
                    partionLength_AF_N -= nLength;
                    goto AVOID_MEM_OVERFLOW;
                }
            }
            return partionLength_AF_N;
        } 

        // 将空间分区 (向下32对齐)
        Int partionLength32() const {
            return this->partionLength_AF_N<32>();
        }

        // 将空间分区 (向下256对齐)
        Int partionLength256() const {
            return this->partionLength_AF_N<256>();
        }

    private:
        List<Queue> pool;                     // 队列池
        List<ExSpaceConsumer> exConsumerList; // 空间分配器列表
        Int alignSize = static_cast<uint32_t>0xffffffff; // 分区时要对齐的字节数
        Int totalSize;                        // 可用空间
        bool doubleBufferEnabled = false;     // 计算DoubleBuffer与否
    };

    // 核内切分策略
    struct CoreTiling {
        Int num;                           // 应用该策略的核心数目
        Int batchNum;                      // 批次数目
        Int batchPartitionLength;          // 批次分区长度 (分区是一个度量 各队列搬入&搬出&长度&运算量的基本变量)
        Int formerTileNum;                 // 大Tile数量
        Int formerTilePartitionLength;     // 大Tile分区长度
        Int tailTilePartitionLength;       // 小Tile分区长度
        Int workload;       // 估算的工作量
    };
    
    using kunlun::tiling::sokmak;
    using kunlun::tiling::sokmakWithWorkload;
    using WorkloadFunc = kunlun::tiling::WorkloadFunc;
    
    // 简单的 Tiling切分器 
    // (按长度比重&数据类型切分)
    // (支持32B/256B对齐)
    // (支持多核)
    // (使用比重队列的前提: 各 队列 总长 必须能被 长度比重 整除 如:[总长40,比重2] or 即使 输入 冗余复制&计算 也不影响结果 )
    // (Input数据类型大小仅支持 1、2、4、8)
    // (如某些IO无法使用比重队列，则可以转为使用 ExSpaceConsumer处理额外空间消耗 )
    class SimpleTilingStrategy {
    public:
        CoreDetail coreDetail;      // 核心详情
        CoreTiling formerCore;      // 大核切分策略
        CoreTiling tailCore;        // 小核切分策略
        List<Queue> inputs;         // 输入队列
        List<Queue> calcs;          // 临时变量
        List<Queue> outputs;        // 输出队列
        Int fc_stdLen;              // 大核 标准处理总长
        Int tc_stdLen;              // 小核 标准处理总长
        // 构造
        explicit SimpleTilingStrategy(const gert::TilingContext* context, const platform_ascendc::CoreMemType coreMemType) : 
            workload(nullptr) {
            this->context = context;
            // 解析 输入输出
            parseIOQueue();
            // 解析 硬件详情
            parseCoreDetail(coreMemType);
            // 给出默认值
            this->coreDetail.coreNum = 0;
            this->QueueInfered  =false;
        }

        // 析构
        ~SimpleTilingStrategy() = default;

        // 新增 计算队列 
        void addCalc(const Int& lengthWeight, const Int& dtSize) { 
            this->calcs.emplace_back(Queue{Position::CALC, lengthWeight, dtSize, 0, 0, 0, 0, 0, 0, 0}); 
        }

        // 设置核数
        void SetBlockDim(Int blockDim) { 
            this->coreDetail.coreNum = blockDim; 
        }

        // 注册 额外 空间 分配器
        void registerExtraSpaceConsumer(ExSpaceConsumer consumer) { 
            this->queuePool.registerExtraSpaceConsumer(consumer); 
        }

        // 注册一个工作量估算函数；如果未指定，则使用默认估算方式（字节最长的IO，每32B数据为一单位工作量）
        // workload: A Int(Int) function, pass an integer X as data-length, return workload of X.
        void registerWorkloadFunc(WorkloadFunc workload) { 
            this->workload = workload; 
        }

        // 重新切分 (默认32B对齐，可手动指定对齐值)
        template<bool DoubleBufferEnable, Int AlignWith = 32> 
        void reTiling() {
            // 入口 检查
            CHECK(checkIOQueueWeight(), "SimpleTilingStrategy::reTiling", "At least one of IO-Queues' lengthWeight is not set.");
            CHECK(this->coreDetail.coreNum > 0, "SimpleTilingStrategy::reTiling", "Core num should be greater than 0.");

            this->QueueInfered  =false;

            // 查找 被切糕的指标 (这是存在风险的一步, 当不满足使用该 Tiling切分策略类 的约束时)
            auto stdLenDetail = getStdLength_max();
            // auto stdLenDetail = getStdLength_first();
            // std::cout << "stdLength: " << std::get<0>(stdLenDetail) << ", powad: " << std::get<1>(stdLenDetail) << ", dtSizeMin: " << std::get<2>(stdLenDetail) << std::endl;
            auto& stdLength = std::get<0>(stdLenDetail);
            auto& powad = std::get<1>(stdLenDetail);
            auto& dtSizeMin = std::get<2>(stdLenDetail);

            // 如果未注册，则使用默认工作量估算方式
            if (!this->workload) {
                this->workload = [powad](Int taskLen) {
                    return (taskLen * powad + 31) / 32;
                };
            }

            // 更新 输出 总长 (这是存在风险的一步, 当不满足使用该 Tiling切分策略类 的约束时)
            updateOutputQueueWithStdLength(stdLength);

            // 分核 切糕
            auto result = sokmakWithWorkload(stdLength, coreDetail.coreNum, dtSizeMin, AlignWith, this->workload);
            Int fl = std::get<0>(result);
            Int fn = std::get<1>(result);
            Int tl = std::get<2>(result);
            Int flWorkload = std::get<3>(result);
            Int tlWorkload = std::get<4>(result);

            // 更新相关数据
            this->fc_stdLen = fl;
            this->tc_stdLen = tl;

            // 每核的核内切分
            auto& pool = this->queuePool; 
            InitQueuePool(pool);
            pool.setDoubleBufferEnable(DoubleBufferEnable);
            auto pLengh = pool.partionLength_AF_N<AlignWith>();

            // 大核
            {
                // 更新相关数据
                auto& core = this->formerCore;
                core.num = fn;
                core.batchNum = 1;
                core.workload = flWorkload;

                // 核内 Tile 信息
                auto ftLength = std::min(pLengh, fl);
                auto ftNum = fl / ftLength;
                auto ttLength = fl % ftLength;

                // 更新相关数据
                core.batchPartitionLength = fl;
                core.formerTileNum = ftNum;
                core.formerTilePartitionLength = ftLength;
                core.tailTilePartitionLength = ttLength;
            }

            // 小核(尾核)
            if (tl > 0) {
                // 更新相关数据
                auto& core = this->tailCore;
                core.num = 1;
                core.batchNum = 1;
                core.workload = tlWorkload;

                // 核内 Tile 信息
                auto ftLength = std::min(pLengh, tl);
                auto ftNum = tl / ftLength;
                auto ttLength = tl % ftLength;

                // 更新相关数据
                core.batchPartitionLength = tl;
                core.formerTileNum = ftNum;
                core.formerTilePartitionLength = ftLength;
                core.tailTilePartitionLength = ttLength;
            }else{
                // 更新相关数据
                auto& core = this->tailCore;
                core.num = 0;
                core.batchNum = 0;
                core.batchPartitionLength = 0;
                core.formerTileNum = 0;
                core.formerTilePartitionLength = 0;
                core.tailTilePartitionLength = 0;
                core.workload = 0;
            }

        }

        // 带指定Batch和BatchLength的重新切分 (Warning: 仅支持IO长度相等(lengthWeight相同)，且均满足Batch和BatchLength的约束)
        template<bool DoubleBufferEnable, Int AlignWith = 1> 
        void reTilingWithBatch(const Int& batchNum, const Int& batchLength) {
            // 入口 检查
            CHECK(checkIOQueueWeightWithBatch(), "SimpleTilingStrategy::reTilingWithBatch", "At least one of IO-Queues' lengthWeight is not same as others'.");
            CHECK(this->coreDetail.coreNum > 0, "SimpleTilingStrategy::reTilingWithBatch", "Core num should be greater than 0.");

            this->QueueInfered  =false;

            // 寻找 dtsize的最值
            Int dtSizeMax = 0;
            for (auto& x : this->inputs) {
                dtSizeMax = std::max(x.dtSize, dtSizeMax);
            }
            for (auto& x : this->outputs) {
                dtSizeMax = std::max(x.dtSize, dtSizeMax);
            }

            // 如果未注册，则使用默认工作量估算方式
            if (!this->workload) {
                this->workload = [batchLength, dtSizeMax](Int taskBatchNum) {
                    return (taskBatchNum * batchLength * dtSizeMax + 31) / 32;
                };
            }

            // 更新 输出 总长 (这是存在风险的一步, 当不满足使用该 Tiling切分策略类 的约束时)
            updateOutputQueueWithStdLength(batchNum * batchLength);

            // 分核 切糕
            auto result = sokmakWithWorkload(batchNum, coreDetail.coreNum, 1, 1, this->workload);
            Int fl = std::get<0>(result);
            Int fn = std::get<1>(result);
            Int tl = std::get<2>(result);
            Int flWorkload = std::get<3>(result);
            Int tlWorkload = std::get<4>(result);

            // 更新相关数据
            this->fc_stdLen = fl*batchLength;
            this->tc_stdLen = tl*batchLength;

            // 核内切分
            auto& pool = this->queuePool;
            InitQueuePool(pool);
            pool.setDoubleBufferEnable(DoubleBufferEnable);
            auto pLengh = pool.partionLength_AF_N<AlignWith>();

            // 大核
            {
                // 更新相关数据
                auto& core = this->formerCore;
                core.num = fn;
                core.batchNum = fl;
                core.batchPartitionLength = batchLength;
                core.workload = flWorkload;

                // 核内 Tile 信息
                auto ftLength = pLengh;
                auto ftNum = batchLength / ftLength;
                auto ttLength = batchLength % ftLength;

                // 更新相关数据
                core.formerTileNum = ftNum;
                core.formerTilePartitionLength = ftLength;
                core.tailTilePartitionLength = ttLength;
            }

            // 小核(尾核)
            if (tl > 0) {
                // 更新相关数据
                auto& core = this->tailCore;
                core.num = 1;
                core.batchNum = tl;
                core.batchPartitionLength = batchLength;
                core.workload = tlWorkload;

                // 核内 Tile 信息
                auto ftLength = pLengh;
                auto ftNum = batchLength / ftLength;
                auto ttLength = batchLength % ftLength;

                // 更新相关数据
                core.formerTileNum = ftNum;
                core.formerTilePartitionLength = ftLength;
                core.tailTilePartitionLength = ttLength;
            }else{
                // 更新相关数据
                auto& core = this->tailCore;
                core.num = 0;
                core.batchNum = 0;
                core.batchPartitionLength = 0;
                core.formerTileNum = 0;
                core.formerTilePartitionLength = 0;
                core.tailTilePartitionLength = 0;
                core.workload = 0;
            }
        }

        // 自动推导 队列详情
        void autoInferQueueDetail(){
            for(auto& I : this->inputs){
                const auto& w = I.lengthWeight;
                I.formercoreLen = this->fc_stdLen * w;
                I.fc_formerTileLen = this->formerCore.formerTilePartitionLength * w;
                I.fc_tailTileLen = this->formerCore.tailTilePartitionLength * w;
                I.tailcoreLen = this->tc_stdLen * w;
                I.tc_formerTileLen = this->tailCore.formerTilePartitionLength * w;
                I.tc_tailTileLen = this->tailCore.tailTilePartitionLength * w;
            }   
            for(auto& O : this->outputs){
                const auto& w = O.lengthWeight;
                O.formercoreLen = this->fc_stdLen * w;
                O.fc_formerTileLen = this->formerCore.formerTilePartitionLength * w;
                O.fc_tailTileLen = this->formerCore.tailTilePartitionLength * w;
                O.tailcoreLen = this->tc_stdLen * w;
                O.tc_formerTileLen = this->tailCore.formerTilePartitionLength * w;
                O.tc_tailTileLen = this->tailCore.tailTilePartitionLength * w;
            }
            for(auto& C : this->calcs){
                const auto& w = C.lengthWeight;
                C.formercoreLen = this->fc_stdLen * w;
                C.fc_formerTileLen = this->formerCore.formerTilePartitionLength * w;
                C.fc_tailTileLen = this->formerCore.tailTilePartitionLength * w;
                C.tailcoreLen = this->tc_stdLen * w;
                C.tc_formerTileLen = this->tailCore.formerTilePartitionLength * w;
                C.tc_tailTileLen = this->tailCore.tailTilePartitionLength * w;
            }
            this->QueueInfered  =true;
        }

        friend std::ostream& operator<<(std::ostream& out, const SimpleTilingStrategy& o);
    protected:
        const gert::TilingContext* context; // Tiling上下文
        QueuePool queuePool;                // 队列池
        WorkloadFunc workload;              // 工作量估算函数
        bool QueueInfered;                  // 队列详情是否已推导
        // 解析 输入输出
        void parseIOQueue() {
            const auto compute_node_info = context->GetComputeNodeInfo();
            const auto& inputsNum = compute_node_info->GetInputsNum();
            const auto& outputsNum = compute_node_info->GetOutputsNum();
            for (auto i = 0; i < inputsNum; ++i) {
                const auto& shape = context->GetInputShape(i)->GetStorageShape();
                Int dtSize;
                ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(i)->GetDataType(), dtSize);
                Queue que = { Position::INPUT, 1, dtSize, Int(shape.GetShapeSize()), 0, 0, 0, 0, 0, 0};
                inputs.emplace_back(que);
            }
            for(auto i = 0; i < outputsNum; ++i) {
                // const auto& shape = context->GetOutputShape(i)->GetStorageShape();
                Int dtSize;
                ge::TypeUtils::GetDataTypeLength(context->GetOutputDesc(i)->GetDataType(), dtSize);
                Queue que = { Position::OUTPUT, 1, dtSize, INT_MAX, 0, 0, 0, 0, 0, 0};
                outputs.emplace_back(que);
            }
        }

        // 解析 核 详情
        void parseCoreDetail(const platform_ascendc::CoreMemType coreMemType) {
            // 获取核内空间大小
            uint64_t memSize;
            platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            ascendcPlatform.GetCoreMemSize(coreMemType, memSize);
            this->coreDetail.memSize = memSize;
        }

        // 检查是否 所有IO的lengthWeight 都被设置了
        bool checkIOQueueWeight() const {
            for (auto& I : this->inputs) {
                if (I.lengthWeight == INT_MAX) {
                    return false;
                }
            }
            for (auto& O : this->outputs) {
                if (O.lengthWeight == INT_MAX) {
                    return false;
                }
            }
            return true;
        }

        // 检查是否 所有IO的lengthWeight 都被设置了 同一个值
        bool checkIOQueueWeightWithBatch() const {
            Int weight = this->inputs[0].lengthWeight;
            for (auto& I : this->inputs) {
                if (I.lengthWeight != weight) {
                    return false;
                }
            }
            for (auto& O : this->outputs) {
                if (O.lengthWeight != weight) {
                    return false;
                }
            }
            return true;
        }

        // 查找 被切糕的指标
        // Exist [totalLength, dtSize] for x in [inque1,inque2...] meet max of 
        // weight = max( x.lengthWeight , stdLen ) 
        std::tuple<Int, Int, Int> getStdLength_max() const {
            Int stdLenPerWeightMax = 0;
            Int powadMax = 0; // Max product of weight and dtSize.
            Int dtSizeMin = Int(0) - 1;
            for (auto& x : this->inputs) {
                stdLenPerWeightMax = std::max(x.totalLength / x.lengthWeight, stdLenPerWeightMax);
                dtSizeMin = std::min(x.dtSize, dtSizeMin);
                powadMax = std::max(x.lengthWeight * x.dtSize, powadMax);
            }
            return std::make_tuple(stdLenPerWeightMax, powadMax, dtSizeMin);
        }
        
        // // x0 = arr([inque1,que2...]).get(0)
        // // stdLen = x.totalLength 
        // std::tuple<Int, Int> getStdLength_first() const{
        //     auto& x0 = this->inputs[0];
        //     return std::make_tuple(x0.totalLength/x0.lengthWeight, x0.lengthWeight);
        // }

        // 根据标准长度更新输出队列总长
        void updateOutputQueueWithStdLength(Int stdLength) {
            for (auto& output : outputs) {
                output.totalLength = output.lengthWeight * stdLength;
            }
        }

        // 初始化队列池
        void InitQueuePool(QueuePool& pool) {
            // 清空队列池
            pool.clear();
            // 设置参数
            pool.setTotalSize(coreDetail.memSize);
            for (auto& I : this->inputs) pool.add(I);
            for (auto& C : this->calcs) pool.add(C);
            for (auto& O : this->outputs) pool.add(O);
        }
    };

    // 允许标准化输出到std
    std::ostream& operator<<(std::ostream& out, const SimpleTilingStrategy& o){
        out << "SimpleTilingStrategy: {\n";
            out << "\tCore Detail: {\n\t\tcoreNum: " << o.coreDetail.coreNum << ",\n\t\tmemSize: " << o.coreDetail.memSize << "\n\t}\n";
            out << "\tFormer Core: {\n\t\tnum: " << o.formerCore.num 
                << ",\n\t\tbatchNum: " << o.formerCore.batchNum 
                << ",\n\t\tbatchPartitionLength: " << o.formerCore.batchPartitionLength 
                << ",\n\t\tformerTileNum: " << o.formerCore.formerTileNum 
                << ",\n\t\tformerTilePartitionLength: " << o.formerCore.formerTilePartitionLength 
                << ",\n\t\ttailTilePartitionLength: " << o.formerCore.tailTilePartitionLength 
                << ",\n\t\tworkload: " << o.formerCore.workload 
                << "\n\t}\n";
            out << "\tTail Core: {\n\t\tnum: " << o.tailCore.num 
                << ",\n\t\tbatchNum: " << o.tailCore.batchNum 
                << ",\n\t\tbatchPartitionLength: " << o.tailCore.batchPartitionLength 
                << ",\n\t\tformerTileNum: " << o.tailCore.formerTileNum 
                << ",\n\t\tformerTilePartitionLength: " << o.tailCore.formerTilePartitionLength 
                << ",\n\t\ttailTilePartitionLength: " << o.tailCore.tailTilePartitionLength 
                << ",\n\t\tworkload: " << o.tailCore.workload 
                << "\n\t}\n";
            out << "\tInputs: [\n";
            for (auto i = 0 ; i < o.inputs.size(); ++i) {
                const auto& input = o.inputs[i];
                out << "\t\tId-"<<i
                    <<":{\n\t\t\tlengthWeight: " << input.lengthWeight 
                    << ",\n\t\t\tdtSize: " << input.dtSize 
                    << ",\n\t\t\ttotalLength: " << input.totalLength ;
                if(o.QueueInfered){
                    out << "\n\t\t\tformerCore:{" 
                    << "\n\t\t\t\tprocessLength: " << input.formercoreLen
                    << "\n\t\t\t\tformerLength: " << input.fc_formerTileLen
                    << "\n\t\t\t\ttailLength: " << input.fc_tailTileLen
                    << "\n\t\t\t}"
                    << "\n\t\t\ttailCore:{" 
                    << "\n\t\t\t\tprocessLength: " << input.tailcoreLen
                    << "\n\t\t\t\tformerLength: " << input.tc_formerTileLen
                    << "\n\t\t\t\ttailLength: " << input.tc_tailTileLen
                    << "\n\t\t\t}\n\t\t";
                }
                out << " },\n";
            }
            out << "\t]\n";
            out << "\tOutputs: [\n";
            for (auto i = 0 ; i < o.outputs.size(); ++i) {
                const auto& output = o.outputs[i];
                out << "\t\tId-"<<i
                    <<":{\n\t\t\tlengthWeight: " << output.lengthWeight 
                    << ",\n\t\t\tdtSize: " << output.dtSize 
                    << ",\n\t\t\ttotalLength: " << output.totalLength ;
                if(o.QueueInfered){
                    out << "\n\t\t\tformerCore:{" 
                    << "\n\t\t\t\tprocessLength: " << output.formercoreLen
                    << "\n\t\t\t\tformerLength: " << output.fc_formerTileLen
                    << "\n\t\t\t\ttailLength: " << output.fc_tailTileLen
                    << "\n\t\t\t}"
                    << "\n\t\t\ttailCore:{" 
                    << "\n\t\t\t\tprocessLength: " << output.tailcoreLen
                    << "\n\t\t\t\tformerLength: " << output.tc_formerTileLen
                    << "\n\t\t\t\ttailLength: " << output.tc_tailTileLen
                    << "\n\t\t\t}\n\t\t";
                }
                out << " },\n";
            }
            out << "\t]\n";
            out << "\tCalcs: [\n";
            for (auto i = 0 ; i < o.calcs.size(); ++i) {
                const auto& calc = o.calcs[i];
                out << "\t\tId-"<<i
                    <<":{\n\t\t\tlengthWeight: " << calc.lengthWeight 
                    << ",\n\t\t\tdtSize: " << calc.dtSize 
                    << ",\n\t\t\tformerCoreLength: " << calc.formercoreLen 
                    << ",\n\t\t\ttailCoreLength: " << calc.tailcoreLen ;
                out << " },\n";
            }
            out << "\t]\n";
        out << "}\n";
        return out;
    }
} // namespace tiling
} // namespace kunlun

#endif// TILING_KUNLUN_H

