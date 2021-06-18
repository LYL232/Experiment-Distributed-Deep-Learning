//
// Created by LYL232 on 2021/2/16.
//

#include "global/Global.h"
#include "communicate/tensor/end2end/controller/bcc/BlockedEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

BlockedEnd2EndCommunicateController::BlockedEnd2EndCommunicateController(
        std::shared_ptr<CommunicationBackend> backend,
        std::shared_ptr<BlockedEnd2EndCommunication> communication
) : TensorEnd2EndCommunicateController(std::move(backend)), communicationImpl_(std::move(communication)) {}

StatusCode
BlockedEnd2EndCommunicateController::handleRequest(const std::shared_ptr<TensorEnd2EndCommunicateRequest> &request) {
    // 由调用Op本身的线程去阻塞执行这个请求, 而不是像rtc的组通信那样交给后台线程统一执行
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_TF_OP_INTERACTION
    using namespace std;
    string requestTypeName;
    if (dynamic_cast<TensorSendCommunicateRequest *>(request.get()) != nullptr) {
        requestTypeName = "send: ";
    } else if (dynamic_cast<TensorReceiveCommunicateRequest *>(request.get()) != nullptr) {
        requestTypeName = "receive: ";
    } else {
        requestTypeName = "unknown: ";
    }
    GLOBAL_INFO_WITH_THREAD_ID(
            "op request to " << requestTypeName << "tensor: " << request->key() <<
                             ", tag:" << request->tag())
#endif
    return request->end2EndCommunicate();
}

StatusCode BlockedEnd2EndCommunicateController::send(const TensorSendCommunicateRequest &request) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_COMMUNICATE
    GLOBAL_INFO_WITH_THREAD_ID("sending Tensors: " << request.key())
#endif
    return communicationImpl_->send(request);
}

StatusCode BlockedEnd2EndCommunicateController::receive(const TensorReceiveCommunicateRequest &request) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_COMMUNICATE
    GLOBAL_INFO_WITH_THREAD_ID("receiving Tensors: " << request.key())
#endif
    return communicationImpl_->receive(request);
}

}}}}