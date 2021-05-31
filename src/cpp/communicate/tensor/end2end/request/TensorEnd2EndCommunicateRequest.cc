//
// Created by LYL232 on 2021/2/16.
//

#include "def.h"
#include "communicate/tensor/end2end/request/TensorEnd2EndCommunicateRequest.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        TensorEnd2EndCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<CommonTensor> requestingTensor,
        std::function<void(StatusCode)> done,
        std::shared_ptr<Communicator> communicator,
        std::shared_ptr<OpContext> context) :
        TensorCommunicateRequest(key, std::move(requestingTensor), std::move(done),
                                 std::move(communicator), std::move(context)),
        controller_(controller) {}

TensorEnd2EndCommunicateRequest::TensorEnd2EndCommunicateRequest(
        TensorEnd2EndCommunicateRequest &&other) noexcept:
        TensorCommunicateRequest(std::move(other)), controller_(other.controller_) {}

StatusCode TensorEnd2EndCommunicateRequest::end2EndCommunicate() {
    CALLING_ABSTRACT_INTERFACE_ERROR("TensorEnd2EndCommunicateRequest::end2EndCommunicate()");
}

}}}