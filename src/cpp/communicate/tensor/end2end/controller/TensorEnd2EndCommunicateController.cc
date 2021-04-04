//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorEnd2EndCommunicateController::TensorEnd2EndCommunicateController(
        std::shared_ptr<CommunicationBackend> backend) : backend_(std::move(backend)) {}

StatusCode TensorEnd2EndCommunicateController::handleRequest
        (const std::shared_ptr<TensorEnd2EndCommunicateRequest> &) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorEnd2EndCommunicateController::"
            "handleRequest(const std::shared_ptr<TensorEnd2EndCommunicateRequest> &)");
}

StatusCode TensorEnd2EndCommunicateController::send(
        const TensorSendCommunicateRequest &request) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorEnd2EndCommunicateController::"
            "send(const Request &request)");
}

StatusCode TensorEnd2EndCommunicateController::receive(
        const TensorReceiveCommunicateRequest &request) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorEnd2EndCommunicateController::"
            "receive(const Request &request)");
}

}}}