//
// Created by LYL232 on 2021/2/12.
//

#include "communicate/tensor/collective/broadcast/TensorBroadcastRequest.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

const char *TensorBroadcastRequest::requestType = "Broadcast";

TensorBroadcastRequest::TensorBroadcastRequest(
        TensorsCollectiveCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::shared_ptr<tensorflow::Tensor> resultTensor,
        std::function<void(StatusCode)> done,
        int rootRank) :
        TensorCollectiveCommunicateRequest(
                controller, key, requestingTensor, resultTensor, done
        ), rootRank_(rootRank) {}

TensorBroadcastRequest::TensorBroadcastRequest(const TensorBroadcastRequest &other) :
        TensorCollectiveCommunicateRequest(other) {}

TensorBroadcastRequest::TensorBroadcastRequest(TensorBroadcastRequest &&other) :
        TensorCollectiveCommunicateRequest(std::move(other)) {
}

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