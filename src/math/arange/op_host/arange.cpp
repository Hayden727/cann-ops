
#include "arange_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
    const uint32_t UNIT_NUM_SIZE = 256;
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ArangeTilingData tiling;
        uint32_t totalLength = context->GetOutputShape(0)->GetOriginShape().GetShapeSize();
        ge::DataType dtype_out = context->GetOutputDesc(0)->GetDataType();
        uint32_t dtype_size = 2;
        context->SetTilingKey(0);
        if (dtype_out == ge::DataType::DT_FLOAT16)
        {
            dtype_size = 2;
        }
        else if (dtype_out == ge::DataType::DT_BF16)
        {
            dtype_size = 2;
        }
        else if (dtype_out == ge::DataType::DT_FLOAT)
        {
            dtype_size = 4;
            context->SetTilingKey(1);
        }
        else if (dtype_out == ge::DataType::DT_INT32)
        {
            dtype_size = 4;
        }
        else if (dtype_out == ge::DataType::DT_INT64)
        {
            dtype_size = 8;
        }
        else
        {
            return ge::GRAPH_FAILED;
        }

        uint32_t totalNum = totalLength;
        uint32_t unitNum  = UNIT_NUM_SIZE / dtype_size;
        uint32_t unitLoops = totalNum / unitNum;
        uint32_t tailNum = totalNum - unitNum * unitLoops;
        if( tailNum > 0 ) unitLoops += 1;

        tiling.set_dtypeSize(dtype_size);
        tiling.set_totalNum(totalNum);
        tiling.set_unitNum(unitNum);
        tiling.set_unitLoops(unitLoops);
        tiling.set_tailNum(tailNum);

        // printf("totalNum %d unitNum %d tailNum %d loops\n",
        //     totalNum, unitNum, tailNum, unitLoops);

        context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Arange : public OpDef
    {
    public:
        explicit Arange(const char *name) : OpDef(name)
        {
            this->Input("start")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .Scalar();
            this->Input("end")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .Scalar();
            this->Input("step")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .Scalar();
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend310b");
        }
    };

    OP_ADD(Arange);
}
