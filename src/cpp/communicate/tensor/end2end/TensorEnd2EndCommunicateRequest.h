//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSOREND2ENDCOMMUNICATEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSOREND2ENDCOMMUNICATEREQUEST_H

#include "communicate/tensor/TensorCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorEnd2EndCommunicateController;

class TensorEnd2EndCommunicateRequest : public TensorCommunicateRequest {
public:
    TensorEnd2EndCommunicateRequest(
            TensorEnd2EndCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<tensorflow::Tensor> requestingTensor,
            std::function<void(StatusCode)> done
    );

    TensorEnd2EndCommunicateRequest(const TensorEnd2EndCommunicateRequest &other);

    TensorEnd2EndCommunicateRequest(TensorEnd2EndCommunicateRequest &&other);

    virtual StatusCode end2EndCommunicate();

protected:
    TensorEnd2EndCommunicateController &controller_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSOREND2ENDCOMMUNICATEREQUEST_H
