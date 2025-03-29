#include "register/tilingdata_base.h"
#include "./tiling_base.h"
namespace optiling { 
  BEGIN_TILING_DATA_DEF(MoeV2VBSComputeTilingData)
  TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
  TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
  TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
  TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
  TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
  TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
  TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
  TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
  TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(MoeV2VBSComputeTilingDataOp, MoeV2VBSComputeTilingData)
   
  BEGIN_TILING_DATA_DEF(MoeV2VMSMiddleComputeTilingData)
  TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(MoeV2VMSMiddleComputeTilingDataOp, MoeV2VMSMiddleComputeTilingData)

  BEGIN_TILING_DATA_DEF(MoeV2SortOutComputeTilingData)
  TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(MoeV2SortOutComputeTilingDataOp, MoeV2SortOutComputeTilingData)

  BEGIN_TILING_DATA_DEF(MoeV2GatherOutComputeTilingData)
  TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, activateRows);
  TILING_DATA_FIELD_DEF(int64_t, perCoreRows);
  TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopRows);
  TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopRows);
  TILING_DATA_FIELD_DEF(int64_t, lastCoreRows);
  TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopRows);
  TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopRows);
  TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
  TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
  TILING_DATA_FIELD_DEF(int64_t, perLoopCols);
  TILING_DATA_FIELD_DEF(int64_t, lastLoopCols);
  TILING_DATA_FIELD_DEF(int64_t, colLoops);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(MoeV2GatherOutComputeTilingDataOp, MoeV2GatherOutComputeTilingData)

  BEGIN_TILING_DATA_DEF(MoeInitRoutingV2TilingData)
  TILING_DATA_FIELD_DEF(int64_t, coreNum);
  TILING_DATA_FIELD_DEF(int64_t, n);
  TILING_DATA_FIELD_DEF(int64_t, cols);
  TILING_DATA_FIELD_DEF(int64_t, k);
  TILING_DATA_FIELD_DEF(int64_t, expertCapacity);
  TILING_DATA_FIELD_DEF(int64_t, expertNum);
  TILING_DATA_FIELD_DEF(int64_t, dropPadMode);
  TILING_DATA_FIELD_DEF(int64_t, expertTokensCountOrCumsumFlag);
  TILING_DATA_FIELD_DEF(int64_t, expertTokensBeforeCapacityFlag);
TILING_DATA_FIELD_DEF(int64_t, start_expertId);
TILING_DATA_FIELD_DEF(int64_t,end_expertId);
TILING_DATA_FIELD_DEF(int64_t,device_id);
  TILING_DATA_FIELD_DEF_STRUCT(MoeV2VBSComputeTilingData, vbsComputeParamsOp);
  TILING_DATA_FIELD_DEF_STRUCT(MoeV2VMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);
  TILING_DATA_FIELD_DEF_STRUCT(MoeV2SortOutComputeTilingData, sortOutComputeParamsOp);
  TILING_DATA_FIELD_DEF_STRUCT(MoeV2GatherOutComputeTilingData, srcToDstComputeParamsOp);
  TILING_DATA_FIELD_DEF_STRUCT(MoeV2GatherOutComputeTilingData, srcToDstCapacityComputeParamsOp);
  TILING_DATA_FIELD_DEF_STRUCT(MoeV2GatherOutComputeTilingData, gatherOutComputeParamsOp);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2, MoeInitRoutingV2TilingData)
  struct MoeInitRoutingV2CompileInfo {};


class MoeInitRoutingV2TilingBase : public TilingBaseClass {
 public:
  explicit MoeInitRoutingV2TilingBase(gert::TilingContext* context) : TilingBaseClass(context) {
    Reset();
  }
  ~MoeInitRoutingV2TilingBase() override = default;

  void Reset(gert::TilingContext* context) override {
    TilingBaseClass::Reset(context);
    Reset();
  }

 protected:
  bool IsCapable() override {
    return true;
  }
  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
  ge::graphStatus GetPlatformInfo() override;
  // 2、获取INPUT/OUTPUT/ATTR信息
  ge::graphStatus GetShapeAttrsInfo() override;
  // 3、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;
  // 4、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override;
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 6、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;
  void Reset();

 protected:
  ge::graphStatus CheckTokenCount(int64_t num, const char* tag);
  virtual ge::graphStatus CheckOutShape();
  virtual void Tiling4GatherOutCompute();
  void Tiling4SrcToDstCompute();
  virtual void Tiling4SrcToDstCapacityCompute();
  void Tiling4SortOutCompute();
  void Tiling4VMSMiddleCompute();
  void Tiling4VBSCompute();
  void ShowTilingData();
  void Tiling4VBSMultiCoreCompute(MoeV2VBSComputeTilingData* tilingData);
  void Tiling4VBSOneCoreCompute(MoeV2VBSComputeTilingData* tilingData);
  virtual bool IsFullLoad();

  int64_t aivNum;
  int64_t sortLoopMaxElement = 0;
  int64_t mrgSortListMaxElement = 2040;
  int64_t totalLength = 0;
  int64_t activateNum = 0;
  int64_t expertCapacity = 0;
  int64_t expertNum = 0;
  int64_t dropPadMode = 0;

  int64_t start_expertId = 0;
  int64_t end_expertId = 0;
  int64_t device_id = 0;

  int64_t expertTokensCountOrCumsumFlag = 0;
  bool expertTokensBeforeCapacityFlag = false;
  int64_t inuptXDtypeSize_;
  bool isFullLoad = false;

  const char* opName = "";
  MoeInitRoutingV2TilingData moeInitRoutingTilingData;
};

}
