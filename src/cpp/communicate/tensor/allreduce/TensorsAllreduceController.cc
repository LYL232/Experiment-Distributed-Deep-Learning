//
// Created by LYL232 on 2021/2/10.
//

#include "TensorsAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce {

StatusCode TensorsAllreduceController::handleTenorAllreduceRequest(
        const std::string &name,
        std::shared_ptr<tensorflow::Tensor> sendTensor,
        std::shared_ptr<tensorflow::Tensor> recvTensor,
        std::function<void(StatusCode)> done) {
    throw std::runtime_error("trying calling raw handleTenorAllreduceRequest");
}

TensorsAllreduceController::~TensorsAllreduceController() {}
}}}}