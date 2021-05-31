//
// Created by LYL232 on 2021/2/28.
//

#include "communicate/tensor/end2end/request/TensorSendCommunicateRequest.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {


TensorSendCommunicateRequest::TensorSendCommunicateRequest(
        TensorEnd2EndCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<CommonTensor> requestingTensor,
        std::function<void(StatusCode)> done, int receiver,
        std::shared_ptr<Communicator> communicator,
        std::shared_ptr<OpContext> context) :
        TensorEnd2EndCommunicateRequest(
                controller, key, std::move(requestingTensor),
                std::move(done), std::move(communicator), std::move(context)),
        receiver_(receiver) {}

TensorSendCommunicateRequest::TensorSendCommunicateRequest(
        TensorSendCommunicateRequest &&other) noexcept:
        TensorEnd2EndCommunicateRequest(std::move(other)), receiver_(other.receiver_) {}

int TensorSendCommunicateRequest::receiver() const noexcept {
    return receiver_;
}

StatusCode TensorSendCommunicateRequest::end2EndCommunicate() {
    return controller_.send(*this);
}


}}}