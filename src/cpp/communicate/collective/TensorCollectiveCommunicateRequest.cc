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
        : TensorCommunicateRequest(key, requestingTensor, resultTensor, done),
          controller_(controller) {
}

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(
        const TensorCollectiveCommunicateRequest &other) noexcept
        : TensorCommunicateRequest(other),
          controller_(other.controller_) {}

TensorCollectiveCommunicateRequest::TensorCollectiveCommunicateRequest(
        TensorCollectiveCommunicateRequest &&other) noexcept
        : TensorCommunicateRequest(std::move(other)),
          controller_(other.controller_) {}

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

