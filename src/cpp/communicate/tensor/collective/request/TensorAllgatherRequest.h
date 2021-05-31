//
// Created by LYL232 on 2021/2/10.
//

#ifndef EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLGATHERREQUEST_H
#define EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLGATHERREQUEST_H

#include "communicate/backend/CommunicationBackend.h"
#include "communicate/tensor/collective/request/TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorAllgatherRequest : public TensorCollectiveCommunicateRequest {
public:
    using Operation = Communicator::AllreduceOperation;

    TensorAllgatherRequest(
            TensorsCollectiveCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<CommonTensor> requestingTensor,
            std::shared_ptr<CommonTensor> resultTensor,
            std::function<void(StatusCode)> done,
            std::shared_ptr<Communicator> communicator,
            std::shared_ptr<OpContext> context
    );

    TensorAllgatherRequest(const TensorAllgatherRequest &other) = default;

    TensorAllgatherRequest(TensorAllgatherRequest &&other) noexcept;

    StatusCode collectiveCommunicate(const Requests &requests) override;

    const char *requestTypeName() const noexcept override;

    static const char *requestType;
};

}}}


#endif //EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLGATHERREQUEST_H
