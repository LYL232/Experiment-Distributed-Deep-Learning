//
// Created by LYL232 on 2021/2/12.
//

#include "def.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

StatusCode TensorsCollectiveCommunicateController::handleRequest(
        const std::shared_ptr<TensorCollectiveCommunicateRequest> &) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorsCollectiveCommunicateController::"
            "handleRequest(const std::shared_ptr<TensorCollectiveCommunicateRequest> &)");
}

StatusCode
TensorsCollectiveCommunicateController::allreduce(
        const std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>> &requests
) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorsCollectiveCommunicateController::allreduce("
            "const std::queue<std::shared_ptr<TensorCollectiveCommunicateRequest>> &requests)"
    );
}

StatusCode
TensorsCollectiveCommunicateController::allgather(
        const std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>> &requests
) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorsCollectiveCommunicateController::allgather("
            "const std::queue<std::shared_ptr<TensorCollectiveCommunicateRequest>> &requests)"
    );
}

StatusCode
TensorsCollectiveCommunicateController::broadcast(
        const std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>> &requests
) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "TensorsCollectiveCommunicateController::broadcast("
            "const std::queue<std::shared_ptr<TensorCollectiveCommunicateRequest>> &requests)"
    );
}

}}}
