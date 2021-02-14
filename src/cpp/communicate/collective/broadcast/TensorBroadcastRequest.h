//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORBROADCASTREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORBROADCASTREQUEST_H

#include "communicate/communication/CommunicationBackend.h"
#include "communicate/collective/TensorCollectiveCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorBroadcastRequest : public TensorCollectiveCommunicateRequest {
public:
    TensorBroadcastRequest(
            TensorsCollectiveCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<tensorflow::Tensor> requestingTensor,
            std::shared_ptr<tensorflow::Tensor> resultTensor,
            std::function<void(StatusCode)> done,
            int rootRank
    );

    TensorBroadcastRequest(const TensorBroadcastRequest &other);

    TensorBroadcastRequest(TensorBroadcastRequest &&other);

    virtual StatusCode doCollectiveCommunication(const Requests &requests) override;

    virtual const char *requestTypeName() const noexcept override;

    int rootRank() const noexcept;

    static const char *requestType;

private:
    int rootRank_;


};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORBROADCASTREQUEST_H
