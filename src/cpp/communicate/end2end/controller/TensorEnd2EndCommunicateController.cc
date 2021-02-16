//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorEnd2EndCommunicateController::TensorEnd2EndCommunicateController(
        std::shared_ptr<CommunicationBackend> backend) : backend_(backend) {}

StatusCode TensorEnd2EndCommunicateController::handleRequest(std::shared_ptr<TensorEnd2EndCommunicateRequest>) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorEnd2EndCommunicateController::"
            "handleRequest(std::shared_ptr<TensorEnd2EndCommunicateRequest>)");
}

StatusCode TensorEnd2EndCommunicateController::sendOrRecv(
        const TensorEnd2EndCommunicateRequest &request) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorEnd2EndCommunicateController::"
            "sendOrRecv(const Request &request)");
}
}}}