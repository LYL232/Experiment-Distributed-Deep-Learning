//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/tensor/TensorCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorCommunicateRequest::TensorCommunicateRequest(
        std::string key,
        std::shared_ptr<CommonTensor> requestingTensor,
        std::function<void(StatusCode)> done,
        std::shared_ptr<Communicator> communicator,
        std::shared_ptr<OpContext> context)
        : key_(std::move(key)), requestingTensor_(std::move(requestingTensor)), context_(context),
          done_(std::move(done)), communicator_(std::move(communicator)) {}

TensorCommunicateRequest::TensorCommunicateRequest(
        TensorCommunicateRequest &&other) noexcept
        : key_(std::move(other.key_)),
          requestingTensor_(std::move(other.requestingTensor_)), context_(other.context_),
          done_(std::move(other.done_)), communicator_(std::move(other.communicator_)) {}

std::shared_ptr<CommonTensor> &TensorCommunicateRequest::requestingTensor()
const noexcept {
    return requestingTensor_;
}

void TensorCommunicateRequest::done(StatusCode code) const noexcept {
    done_(code);
}

const std::string &TensorCommunicateRequest::key() const noexcept {
    return key_;
}

const std::shared_ptr<Communicator> &TensorCommunicateRequest::communicator() const noexcept {
    return communicator_;
}

std::shared_ptr<OpContext> &TensorCommunicateRequest::context() const {
    return context_;
}

}}}