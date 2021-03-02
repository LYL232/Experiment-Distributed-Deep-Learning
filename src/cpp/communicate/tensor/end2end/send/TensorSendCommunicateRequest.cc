//
// Created by LYL232 on 2021/2/28.
//

#include "communicate/tensor/end2end/send/TensorSendCommunicateRequest.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {


TensorSendCommunicateRequest::TensorSendCommunicateRequest(
        TensorEnd2EndCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::function<void(StatusCode)> done, int receiver) :
        TensorEnd2EndCommunicateRequest(controller, key, requestingTensor, done),
        receiver_(receiver) {}

TensorSendCommunicateRequest::TensorSendCommunicateRequest(
        const TensorSendCommunicateRequest &other) :
        TensorEnd2EndCommunicateRequest(other), receiver_(other.receiver_) {}

TensorSendCommunicateRequest::TensorSendCommunicateRequest(
        TensorSendCommunicateRequest &&other) :
        TensorEnd2EndCommunicateRequest(other), receiver_(other.receiver_) {}

int TensorSendCommunicateRequest::receiver() const noexcept {
    return receiver_;
}

StatusCode TensorSendCommunicateRequest::end2EndCommunicate() {
    return controller_.send(*this);
}


}}}