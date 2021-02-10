//
// Created by LYL232 on 2021/2/10.
//

#include "TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::shared_ptr<tensorflow::Tensor> resultTensor,
        std::function<void(StatusCode)> done)
        : key_(key),
          requestingTensor_(requestingTensor), resultTensor_(resultTensor),
          done_(done) {
    checkDtypeAndNumElements_(requestingTensor, resultTensor);
}

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(const TensorCollectiveCommunicateRequest &other) noexcept
        : key_(other.key_),
          requestingTensor_(other.requestingTensor_), resultTensor_(other.resultTensor_),
          done_(other.done_) {}

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(TensorCollectiveCommunicateRequest &&other) noexcept
        : key_(std::move(other.key_)),
          requestingTensor_(std::move(other.requestingTensor_)),
          resultTensor_(std::move(other.resultTensor_)),
          done_(std::move(other.done_)) {}

std::shared_ptr<tensorflow::Tensor> &TensorCollectiveCommunicateRequest::requestingTensor()
const noexcept {
    return requestingTensor_;
}

std::shared_ptr<tensorflow::Tensor> &TensorCollectiveCommunicateRequest::resultTensor()
const noexcept {
    return resultTensor_;
}

void TensorCollectiveCommunicateRequest::done(StatusCode code) const noexcept {
    done_(code);
}

const std::string &TensorCollectiveCommunicateRequest::key() const noexcept {
    return key_;
}

void TensorCollectiveCommunicateRequest::checkDtypeAndNumElements_(
        const std::shared_ptr<tensorflow::Tensor> &requestingTensor,
        const std::shared_ptr<tensorflow::Tensor> &resultTensor) {
    using namespace std;
    if (requestingTensor->dtype() != resultTensor->dtype()) {
        string msg("try building request with different dtypes: requesting(");
        msg.append(to_string(requestingTensor->dtype())).append("), result(")
                .append(to_string(resultTensor->dtype())).append(")");
        throw runtime_error(msg);
    }
    if (requestingTensor->NumElements() != resultTensor->NumElements()) {
        string msg("try building request with different num elements: requesting(");
        msg.append(to_string(requestingTensor->NumElements())).append("), result(")
                .append(to_string(resultTensor->NumElements())).append(")");
        throw runtime_error(msg);
    }
}

size_t TensorCollectiveCommunicateRequest::tensorSize() const noexcept {
    return requestingTensor_->tensor_data().size();
}

size_t TensorCollectiveCommunicateRequest::elements() const noexcept {
    return requestingTensor_->NumElements();
}

tensorflow::DataType TensorCollectiveCommunicateRequest::dtype() const noexcept {
    return requestingTensor_->dtype();
}

void *TensorCollectiveCommunicateRequest::requestingTensorData() const noexcept {
    return (void *) requestingTensor_->tensor_data().data();
}

void *TensorCollectiveCommunicateRequest::resultTensorData() const noexcept {
    return (void *) resultTensor_->tensor_data().data();
}


}}}

