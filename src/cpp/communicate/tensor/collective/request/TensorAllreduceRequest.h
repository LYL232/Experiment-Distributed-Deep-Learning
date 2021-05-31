//
// Created by LYL232 on 2021/2/10.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLREDUCEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLREDUCEREQUEST_H

#include "communicate/backend/CommunicationBackend.h"
#include "communicate/tensor/collective/request/TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorAllreduceRequest : public TensorCollectiveCommunicateRequest {
public:
    using Operation = Communicator::AllreduceOperation;

    TensorAllreduceRequest(
            TensorsCollectiveCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<CommonTensor> requestingTensor,
            std::shared_ptr<CommonTensor> resultTensor,
            std::function<void(StatusCode)> done,
            Operation op, std::shared_ptr<Communicator> communicator,
            std::shared_ptr<OpContext> context
    );

    TensorAllreduceRequest(const TensorAllreduceRequest &other) = default;

    TensorAllreduceRequest(TensorAllreduceRequest &&other) noexcept;

    Operation op() const noexcept;

    StatusCode collectiveCommunicate(const Requests &requests) override;

    const char *requestTypeName() const noexcept override;

    static const char *requestType;

private:
    Operation op_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLREDUCEREQUEST_H
