//
// Created by LYL232 on 2021/2/16.
//

#include "global/Global.h"
#include "communicate/end2end/controller/bcc/BlockedEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

BlockedEnd2EndCommunicateController::BlockedEnd2EndCommunicateController(
        std::shared_ptr<CommunicationBackend> backend,
        std::shared_ptr<BlockedEnd2EndCommunication> communication
) : TensorEnd2EndCommunicateController(backend), communicationImpl_(communication) {}

StatusCode
BlockedEnd2EndCommunicateController::handleRequest(std::shared_ptr<TensorEnd2EndCommunicateRequest> request) {
    // 由调用Op本身的线程去阻塞执行这个请求, 而不是像rtc的组通信那样交给后台线程统一执行
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_TF_OP_INTERACTION
    using namespace std;
    int rank = backend_->processRank();
    string requestTypeName;
    if (rank == request->sender()) {
        requestTypeName = "send: ";
    } else if (rank == request->receiver()) {
        requestTypeName = "receive: ";
    }
    GLOBAL_INFO_WITH_THREAD_ID(
            "op request to " << requestTypeName << " tensor: " << request->key())
#endif
    return sendOrRecv(*request);
}

StatusCode BlockedEnd2EndCommunicateController::sendOrRecv(const TensorEnd2EndCommunicateRequest &request) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_COMMUNICATE
    using namespace std;
    string desc;
    int rank = backend_->processRank();
    if (rank == request.sender()) {
        desc.append("sending: ");
    } else if (rank == request.receiver()) {
        desc.append("receiving: ");
    }
    desc.append(request.key());
    GLOBAL_INFO_WITH_THREAD_ID("communicating Tensors: " << desc)
#endif
    return communicationImpl_->sendOrReceiveRequest(request);
}

}}}}