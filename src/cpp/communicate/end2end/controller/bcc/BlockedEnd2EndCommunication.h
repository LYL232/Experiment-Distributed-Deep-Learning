//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATION_H

#include "def.h"
#include "communicate/end2end/TensorEnd2EndCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

class BlockedEnd2EndCommunication {
public:
    virtual StatusCode sendOrReceiveRequest(const TensorEnd2EndCommunicateRequest &request) const;

    virtual ~BlockedEnd2EndCommunication() {};
};

}}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATION_H
