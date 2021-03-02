//
// Created by LYL232 on 2021/2/16.
//

#include "def.h"
#include "communicate/tensor/end2end/TensorEnd2EndCommunicateRequest.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        TensorEnd2EndCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::function<void(StatusCode)> done) :
        TensorCommunicateRequest(key, requestingTensor, done),
        controller_(controller) {}

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        const TensorEnd2EndCommunicateRequest &other) :
        TensorCommunicateRequest(other), controller_(other.controller_) {}

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        TensorEnd2EndCommunicateRequest &&other) :
        TensorCommunicateRequest(std::move(other)), controller_(other.controller_) {}

StatusCode TensorEnd2EndCommunicateRequest::end2EndCommunicate() {
    CALLING_ABSTRACT_INTERFACE_ERROR("TensorEnd2EndCommunicateRequest::end2EndCommunicate()");
}

}}}