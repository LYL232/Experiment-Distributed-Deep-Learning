//
// Created by LYL232 on 2021/2/10.
//

#include "global/Global.h"
#include "communicate/collective/TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(
        TensorsCollectiveCommunicateController &controller,
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> requestingTensor,
        std::shared_ptr<tensorflow::Tensor> resultTensor,
        std::function<void(StatusCode)> done)
        : controller_(controller),
          key_(key), requestingTensor_(requestingTensor),
          resultTensor_(resultTensor), done_(done) {
    checkTensorSize_(requestingTensor, resultTensor);
}

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(
        const TensorCollectiveCommunicateRequest &other) noexcept
        : controller_(other.controller_),
          key_(other.key_), requestingTensor_(other.requestingTensor_),
          resultTensor_(other.resultTensor_),
          done_(other.done_) {}

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(
        TensorCollectiveCommunicateRequest &&other) noexcept
        : controller_(other.controller_),
          key_(std::move(other.key_)),
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

void TensorCollectiveCommunicateRequest::checkTensorSize_(
        const std::shared_ptr<tensorflow::Tensor> &requestingTensor,
        const std::shared_ptr<tensorflow::Tensor> &resultTensor) {
    using namespace std;
    if (requestingTensor->tensor_data().size() != resultTensor->tensor_data().size()) {
        string msg("try building request with different sizes: requesting(");
        msg.append(to_string(requestingTensor->tensor_data().size())).append("), result(")
                .append(to_string(resultTensor->tensor_data().size())).append(")");
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

StatusCode TensorCollectiveCommunicateRequest::doCollectiveCommunication(const Requests &requests) {
    // 抽象接口, 当需要进行组通信时由controller调用此函数, 通过虚函数实现多态性: 不同的request调用不同的
    // controller实现的组通信函数, requests包含this
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorCollectiveCommunicateRequest::doCollectiveCommunication("
            "const std::queue<std::shared_ptr<TensorCollectiveCommunicateRequest>> &requests"
    );
}

const char *TensorCollectiveCommunicateRequest::requestTypeName() const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorCollectiveCommunicateRequest::requestTypeName()"
    );
}


}}}

