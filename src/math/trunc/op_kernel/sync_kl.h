/* 
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef SYNC_KUNLUN_H
#define SYNC_KUNLUN_H

/// 跨流水同步函数。通过阻塞DST_POS流水，保证PIPE_SRC的前序读写操作完成后，PIPE_DST上的操作才被执行。
/// 支持的流水：MTE1, MTE2, MTE3, V, S, M
/// @param PIPE_SRC 源流水
/// @param PIPE_DST 目的流水
#define Sync(PIPE_SRC,PIPE_DST) {\
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::PIPE_SRC##_##PIPE_DST)); \
    TQueSync<PIPE_##PIPE_SRC,PIPE_##PIPE_DST> queSync; \
    queSync.SetFlag(eventID); \
    queSync.WaitFlag(eventID); \
}

#endif// SYNC_KUNLUN_H
