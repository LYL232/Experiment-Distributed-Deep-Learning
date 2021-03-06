//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATION_H

#include "def.h"
#include "communicate/tensor/end2end/request/TensorSendCommunicateRequest.h"
#include "communicate/tensor/end2end/request/TensorReceiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

class BlockedEnd2EndCommunication {
public:
    virtual StatusCode send(const TensorSendCommunicateRequest &request) const;

    virtual StatusCode receive(const TensorReceiveCommunicateRequest &request) const;

    virtual ~BlockedEnd2EndCommunication() = default;
protected:
    static std::shared_ptr<HeapMemoryManager> memManager_;
};

}}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATION_H
