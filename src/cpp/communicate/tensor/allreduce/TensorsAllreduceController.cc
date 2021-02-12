//
// Created by LYL232 on 2021/2/10.
//

#include "TensorsAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce {

TensorsAllreduceController::TensorsAllreduceController(
        std::shared_ptr<CommunicationBackend> backend) : backend_(backend) {}


StatusCode TensorsAllreduceController::handleTenorAllreduceRequest(
        const std::string &name,
        std::shared_ptr<tensorflow::Tensor> sendTensor,
        std::shared_ptr<tensorflow::Tensor> recvTensor,
        std::function<void(StatusCode)> done,
        Operation op) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "trying calling abstract interface: TensorsAllreduceController::handleTenorAllreduceRequest"
    );
}

bool TensorsAllreduceController::initialized() const {
    throw std::runtime_error(
            "trying calling abstract interface: TensorsAllreduceController::initialized"
    );
}

TensorsAllreduceController::~TensorsAllreduceController() {}


}}}}