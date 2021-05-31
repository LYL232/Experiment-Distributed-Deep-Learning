//
// Created by LYL232 on 2021/2/10.
//

#include "communicate/tensor/collective/request/TensorAllgatherRequest.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

const char *TensorAllgatherRequest::requestType = "Allgather";

TensorAllgatherRequest::TensorAllgatherRequest(
        TensorsCollectiveCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<CommonTensor> requestingTensor,
        std::shared_ptr<CommonTensor> resultTensor,
        std::function<void(StatusCode)> done,
        std::shared_ptr<Communicator> communicator,
        std::shared_ptr<OpContext> context) :
        TensorCollectiveCommunicateRequest(
                controller, key, std::move(requestingTensor),
                std::move(resultTensor), std::move(done), std::move(communicator),
                std::move(context)
        ) {}

TensorAllgatherRequest::TensorAllgatherRequest(TensorAllgatherRequest &&other) noexcept:
        TensorCollectiveCommunicateRequest(std::move(other)) {}

StatusCode
TensorAllgatherRequest::collectiveCommunicate(const Requests &requests) {
    return controller_.allgather(requests);
}

const char *TensorAllgatherRequest::requestTypeName() const noexcept {
    return requestType;
}

}}}