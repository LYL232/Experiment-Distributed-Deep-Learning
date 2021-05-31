//
// Created by LYL232 on 2021/2/28.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSENDCOMMUNICATEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSENDCOMMUNICATEREQUEST_H

#include "communicate/tensor/end2end/request/TensorEnd2EndCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorSendCommunicateRequest : public TensorEnd2EndCommunicateRequest {
public:
    TensorSendCommunicateRequest(
            TensorEnd2EndCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<CommonTensor> requestingTensor,
            std::function<void(StatusCode)> done,
            int receiver, std::shared_ptr<Communicator> communicator,
            std::shared_ptr<OpContext> context
    );

    TensorSendCommunicateRequest(const TensorSendCommunicateRequest &other) = default;

    TensorSendCommunicateRequest(TensorSendCommunicateRequest &&other) noexcept;

    int receiver() const noexcept;

    StatusCode end2EndCommunicate() override;

private:
    int receiver_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSENDCOMMUNICATEREQUEST_H
