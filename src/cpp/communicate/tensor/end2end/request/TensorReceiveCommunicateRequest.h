//
// Created by LYL232 on 2021/2/28.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORRECEIVECOMMUNICATEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORRECEIVECOMMUNICATEREQUEST_H

#include "communicate/tensor/end2end/request/TensorEnd2EndCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorReceiveCommunicateRequest : public TensorEnd2EndCommunicateRequest {
public:
    TensorReceiveCommunicateRequest(
            TensorEnd2EndCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<CommonTensor> requestingTensor,
            std::function<void(StatusCode)> done,
            int sender, std::shared_ptr<Communicator> communicator,
            std::shared_ptr<OpContext> context
    );

    TensorReceiveCommunicateRequest(const TensorReceiveCommunicateRequest &other) = default;

    TensorReceiveCommunicateRequest(TensorReceiveCommunicateRequest &&other) noexcept;

    int sender() const noexcept;

    StatusCode end2EndCommunicate() override;

private:
    int sender_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORRECEIVECOMMUNICATEREQUEST_H
