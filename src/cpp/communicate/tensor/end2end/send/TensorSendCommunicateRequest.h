//
// Created by LYL232 on 2021/2/28.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSENDCOMMUNICATEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSENDCOMMUNICATEREQUEST_H

#include "communicate/tensor/end2end/TensorEnd2EndCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorSendCommunicateRequest : public TensorEnd2EndCommunicateRequest {
public:
    TensorSendCommunicateRequest(
            TensorEnd2EndCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<tensorflow::Tensor> requestingTensor,
            std::function<void(StatusCode)> done,
            int receiver
    );

    TensorSendCommunicateRequest(const TensorSendCommunicateRequest &other);

    TensorSendCommunicateRequest(TensorSendCommunicateRequest &&other);

    int receiver() const noexcept;

    virtual StatusCode end2EndCommunicate() override;

private:
    int receiver_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSENDCOMMUNICATEREQUEST_H
