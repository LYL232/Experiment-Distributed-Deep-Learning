//
// Created by LYL232 on 2021/2/28.
//

#include "communicate/tensor/end2end/receive/TensorReceiveCommunicateRequest.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"


namespace lyl232 { namespace experiment { namespace ddl {

TensorReceiveCommunicateRequest::TensorReceiveCommunicateRequest(
        TensorEnd2EndCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::function<void(StatusCode)> done, int sender) :
        TensorEnd2EndCommunicateRequest(controller, key, requestingTensor, done),
        sender_(sender) {}

TensorReceiveCommunicateRequest::TensorReceiveCommunicateRequest(
        const TensorReceiveCommunicateRequest &other) :
        TensorEnd2EndCommunicateRequest(other), sender_(other.sender_) {}

TensorReceiveCommunicateRequest::TensorReceiveCommunicateRequest(
        TensorReceiveCommunicateRequest &&other) :
        TensorEnd2EndCommunicateRequest(other), sender_(other.sender_) {}

int TensorReceiveCommunicateRequest::sender() const noexcept {
    return sender_;
}

StatusCode TensorReceiveCommunicateRequest::end2EndCommunicate() {
    return controller_.receive(*this);
}

}}}