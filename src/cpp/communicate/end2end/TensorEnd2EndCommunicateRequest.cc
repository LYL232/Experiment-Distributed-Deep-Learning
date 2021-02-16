//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/end2end/TensorEnd2EndCommunicateRequest.h"
#include "communicate/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        TensorEnd2EndCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::shared_ptr<tensorflow::Tensor> resultTensor,
        std::function<void(StatusCode)> done,
        int sender, int receiver) :
        TensorCommunicateRequest(key, requestingTensor, resultTensor, done),
        controller_(controller), sender_(sender), receiver_(receiver) {}

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        const TensorEnd2EndCommunicateRequest &other) :
        TensorCommunicateRequest(other), controller_(other.controller_),
        sender_(other.sender_), receiver_(other.receiver_) {}

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        TensorEnd2EndCommunicateRequest &&other) :
        TensorCommunicateRequest(std::move(other)), controller_(other.controller_),
        sender_(other.sender_), receiver_(other.receiver_) {}

int TensorEnd2EndCommunicateRequest::sender() const noexcept {
    return sender_;
}

int TensorEnd2EndCommunicateRequest::receiver() const noexcept {
    return receiver_;
}

StatusCode TensorEnd2EndCommunicateRequest::doEnd2EndCommunication() {
    return controller_.sendOrRecv(*this);
}

}}}