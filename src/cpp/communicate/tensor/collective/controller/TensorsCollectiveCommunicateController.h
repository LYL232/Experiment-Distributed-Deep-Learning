//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSCOLLECTIVECOMMUNICATECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSCOLLECTIVECOMMUNICATECONTROLLER_H

#include <vector>
#include "communicate/backend/CommunicationBackend.h"
#include "communicate/tensor/collective/request/TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorsCollectiveCommunicateController {
public:
    using AllreduceOperation = Communicator::AllreduceOperation;
    using Requests = TensorCollectiveCommunicateRequest::Requests;

    TensorsCollectiveCommunicateController() = default;

    TensorsCollectiveCommunicateController(const TensorsCollectiveCommunicateController &) = delete;

    TensorsCollectiveCommunicateController(TensorsCollectiveCommunicateController &&) = delete;

    virtual StatusCode handleRequest(const std::shared_ptr<TensorCollectiveCommunicateRequest> &);

    virtual StatusCode allreduce(const Requests &requests);

    virtual StatusCode allgather(const Requests &requests);

    virtual StatusCode broadcast(const Requests &requests);

    virtual ~TensorsCollectiveCommunicateController() = default;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSCOLLECTIVECOMMUNICATECONTROLLER_H
