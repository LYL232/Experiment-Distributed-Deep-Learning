//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATECONTROLLER_H

#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"
#include "communicate/tensor/end2end/controller/bcc/BlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

class BlockedEnd2EndCommunicateController : public TensorEnd2EndCommunicateController {
public:
    BlockedEnd2EndCommunicateController(
            std::shared_ptr<CommunicationBackend> backend,
            std::shared_ptr<BlockedEnd2EndCommunication> communication
    );

    BlockedEnd2EndCommunicateController(const BlockedEnd2EndCommunicateController &) = delete;

    BlockedEnd2EndCommunicateController(BlockedEnd2EndCommunicateController &&) = delete;

    StatusCode handleRequest(const std::shared_ptr<TensorEnd2EndCommunicateRequest> &request) override;

    StatusCode send(const TensorSendCommunicateRequest &request) override;

    StatusCode receive(const TensorReceiveCommunicateRequest &request) override;

    ~BlockedEnd2EndCommunicateController() override = default;
private:
    std::shared_ptr<BlockedEnd2EndCommunication> communicationImpl_;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKEDEND2ENDCOMMUNICATECONTROLLER_H
