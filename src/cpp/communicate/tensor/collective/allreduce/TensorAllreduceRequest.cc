//
// Created by LYL232 on 2021/2/10.
//

#include "communicate/tensor/collective/allreduce/TensorAllreduceRequest.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

const char *TensorAllreduceRequest::requestType = "Allreduce";

TensorAllreduceRequest::TensorAllreduceRequest(
        TensorsCollectiveCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::shared_ptr<tensorflow::Tensor> resultTensor,
        std::function<void(StatusCode)> done,
        Operation op, std::shared_ptr<Communicator> communicator) :
        TensorCollectiveCommunicateRequest(
                controller, key, std::move(requestingTensor),
                std::move(resultTensor), std::move(done), std::move(communicator)
        ), op_(op) {}

TensorAllreduceRequest::TensorAllreduceRequest(TensorAllreduceRequest &&other) noexcept:
        TensorCollectiveCommunicateRequest(std::move(other)), op_(other.op_) {}

TensorAllreduceRequest::Operation TensorAllreduceRequest::op() const noexcept {
    return op_;
}

StatusCode
TensorAllreduceRequest::collectiveCommunicate(const Requests &requests) {
    return controller_.allreduce(requests);
}

const char *TensorAllreduceRequest::requestTypeName() const noexcept {
    return requestType;
}

}}}