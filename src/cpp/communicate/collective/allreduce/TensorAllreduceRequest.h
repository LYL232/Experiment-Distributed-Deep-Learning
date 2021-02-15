//
// Created by LYL232 on 2021/2/10.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLREDUCEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLREDUCEREQUEST_H

#include "communicate/backend/CommunicationBackend.h"
#include "communicate/collective/TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorAllreduceRequest : public TensorCollectiveCommunicateRequest {
public:
    using Operation = CommunicationBackend::AllreduceOperation;

    TensorAllreduceRequest(
            TensorsCollectiveCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<tensorflow::Tensor> requestingTensor,
            std::shared_ptr<tensorflow::Tensor> resultTensor,
            std::function<void(StatusCode)> done,
            Operation op
    );

    TensorAllreduceRequest(const TensorAllreduceRequest &other);

    TensorAllreduceRequest(TensorAllreduceRequest &&other);

    Operation op() const noexcept;

    virtual StatusCode doCollectiveCommunication(const Requests &requests) override;

    virtual const char *requestTypeName() const noexcept override;

    static const char *requestType;

private:
    Operation op_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORALLREDUCEREQUEST_H
