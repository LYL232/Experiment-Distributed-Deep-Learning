//
// Created by LYL232 on 2021/2/28.
//

#include "communicate/tensor/end2end/request/TensorReceiveCommunicateRequest.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"


namespace lyl232 { namespace experiment { namespace ddl {

TensorReceiveCommunicateRequest::TensorReceiveCommunicateRequest(
        TensorEnd2EndCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<CommonTensor> requestingTensor,
        std::function<void(StatusCode)> done, int sender,
        std::shared_ptr<Communicator> communicator,
        std::shared_ptr<OpContext> context,
        int tag) :
        TensorEnd2EndCommunicateRequest(
                controller, key, std::move(requestingTensor),
                std::move(done), std::move(communicator), std::move(context), tag),
        sender_(sender) {}

TensorReceiveCommunicateRequest::TensorReceiveCommunicateRequest(
        TensorReceiveCommunicateRequest &&other) noexcept:
        TensorEnd2EndCommunicateRequest(std::move(other)), sender_(other.sender_) {}

int TensorReceiveCommunicateRequest::sender() const noexcept {
    return sender_;
}

StatusCode TensorReceiveCommunicateRequest::end2EndCommunicate() {
    return controller_.receive(*this);
}

}}}