
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_KL_DIV_TARGET_BACKWARD_H_
#define ACLNN_KL_DIV_TARGET_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnKlDivTargetBackwardGetWorkspaceSize
 * parameters :
 * gradOutput : required
 * self : required
 * target : required
 * reduction : required
 * logTarget : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
ACLNN_API aclnnStatus aclnnKlDivTargetBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    int64_t reduction,
    bool logTarget,
    aclTensor *gradTarget,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnKlDivTargetBackward
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
ACLNN_API aclnnStatus aclnnKlDivTargetBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
