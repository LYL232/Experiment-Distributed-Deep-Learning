//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/end2end/controller/bcc/BlockedEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

BlockedEnd2EndCommunicateController::BlockedEnd2EndCommunicateController(
        std::shared_ptr<CommunicationBackend> backend,
        std::shared_ptr<BlockedEnd2EndCommunication> communication
) : TensorEnd2EndCommunicateController(backend), communicationImpl_(communication) {}

StatusCode
BlockedEnd2EndCommunicateController::handleRequest(std::shared_ptr<TensorEnd2EndCommunicateRequest> request) {
    // 由调用Op本身的线程去阻塞执行这个请求, 而不是像rtc的组通信那样交给后台线程统一执行
    return sendOrRecv(*request);
}

StatusCode BlockedEnd2EndCommunicateController::sendOrRecv(const TensorEnd2EndCommunicateRequest &request) {
    return communicationImpl_->sendOrReceiveRequest(request);
}


}}}}