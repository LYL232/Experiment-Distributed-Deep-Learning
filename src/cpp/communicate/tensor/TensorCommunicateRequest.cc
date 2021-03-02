//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/tensor/TensorCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorCommunicateRequest::TensorCommunicateRequest(
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::function<void(StatusCode)> done)
        : key_(key), requestingTensor_(requestingTensor), done_(done) {}

TensorCommunicateRequest::TensorCommunicateRequest(
        const TensorCommunicateRequest &other) noexcept
        : key_(other.key_), requestingTensor_(other.requestingTensor_),
          done_(other.done_) {}

TensorCommunicateRequest::TensorCommunicateRequest(
        TensorCommunicateRequest &&other) noexcept
        : key_(std::move(other.key_)),
          requestingTensor_(std::move(other.requestingTensor_)),
          done_(std::move(other.done_)) {}

std::shared_ptr<tensorflow::Tensor> &TensorCommunicateRequest::requestingTensor()
const noexcept {
    return requestingTensor_;
}

void TensorCommunicateRequest::done(StatusCode code) const noexcept {
    done_(code);
}

const std::string &TensorCommunicateRequest::key() const noexcept {
    return key_;
}

size_t TensorCommunicateRequest::tensorSize() const noexcept {
    return requestingTensor_->tensor_data().size();
}

size_t TensorCommunicateRequest::elements() const noexcept {
    return requestingTensor_->NumElements();
}

tensorflow::DataType TensorCommunicateRequest::dtype() const noexcept {
    return requestingTensor_->dtype();
}

void *TensorCommunicateRequest::requestingTensorData() const noexcept {
    return (void *) requestingTensor_->tensor_data().data();
}

}}}