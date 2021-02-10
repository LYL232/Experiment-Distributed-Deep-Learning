//
// Created by LYL232 on 2021/2/10.
//

#include "TensorAllreduceRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorAllreduceRequest::TensorAllreduceRequest(
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::shared_ptr<tensorflow::Tensor> resultTensor,
        std::function<void(StatusCode)> done,
        Operation op) :
        TensorCollectiveCommunicateRequest(key, requestingTensor, resultTensor, done),
        op_(op) {}

TensorAllreduceRequest::TensorAllreduceRequest(const TensorAllreduceRequest &other) :
        TensorCollectiveCommunicateRequest(other), op_(other.op_) {}

TensorAllreduceRequest::TensorAllreduceRequest(TensorAllreduceRequest &&other) :
        TensorCollectiveCommunicateRequest(std::move(other)), op_(other.op_) {
}

TensorAllreduceRequest::Operation TensorAllreduceRequest::op() const noexcept {
    return op_;
}

}}}