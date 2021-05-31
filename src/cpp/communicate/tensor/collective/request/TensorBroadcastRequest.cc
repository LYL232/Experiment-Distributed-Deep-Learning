//
// Created by LYL232 on 2021/2/12.
//

#include "communicate/tensor/collective/request/TensorBroadcastRequest.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

const char *TensorBroadcastRequest::requestType = "Broadcast";

TensorBroadcastRequest::TensorBroadcastRequest(
        TensorsCollectiveCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<CommonTensor> requestingTensor,
        std::shared_ptr<CommonTensor> resultTensor,
        std::function<void(StatusCode)> done,
        int rootRank, std::shared_ptr<Communicator> communicator,
        std::shared_ptr<OpContext> context) :
        TensorCollectiveCommunicateRequest(
                controller, key, std::move(requestingTensor), std::move(resultTensor),
                std::move(done), std::move(communicator), std::move(context)),
        rootRank_(rootRank) {}


TensorBroadcastRequest::TensorBroadcastRequest(TensorBroadcastRequest &&other) noexcept:
        TensorCollectiveCommunicateRequest(std::move(other)), rootRank_(other.rootRank_) {}

StatusCode TensorBroadcastRequest::collectiveCommunicate(const Requests &requests) {
    return controller_.broadcast(requests);
}

int TensorBroadcastRequest::rootRank() const noexcept {
    return rootRank_;
}

const char *TensorBroadcastRequest::requestTypeName() const noexcept {
    return requestType;
}

}}}