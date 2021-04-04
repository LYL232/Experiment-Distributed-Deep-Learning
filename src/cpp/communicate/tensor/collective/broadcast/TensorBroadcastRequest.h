//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORBROADCASTREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORBROADCASTREQUEST_H

#include "communicate/backend/CommunicationBackend.h"
#include "communicate/tensor/collective/TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorBroadcastRequest : public TensorCollectiveCommunicateRequest {
public:
    TensorBroadcastRequest(
            TensorsCollectiveCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<tensorflow::Tensor> requestingTensor,
            std::shared_ptr<tensorflow::Tensor> resultTensor,
            std::function<void(StatusCode)> done,
            int rootRank, std::shared_ptr<Communicator> communicator
    );

    TensorBroadcastRequest(const TensorBroadcastRequest &other) = default;

    TensorBroadcastRequest(TensorBroadcastRequest &&other) noexcept ;

    StatusCode collectiveCommunicate(const Requests &requests) override;

    const char *requestTypeName() const noexcept override;

    int rootRank() const noexcept;

    static const char *requestType;

private:
    int rootRank_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORBROADCASTREQUEST_H
