//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSOREND2ENDCOMMUNICATECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSOREND2ENDCOMMUNICATECONTROLLER_H

#include "communicate/backend/CommunicationBackend.h"
#include "communicate/end2end/TensorEnd2EndCommunicateRequest.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorEnd2EndCommunicateController {
public:
    typedef std::shared_ptr<TensorEnd2EndCommunicateRequest> SharedRequest;

    TensorEnd2EndCommunicateController(std::shared_ptr<CommunicationBackend> backend);

    TensorEnd2EndCommunicateController(const TensorEnd2EndCommunicateController &) = delete;

    TensorEnd2EndCommunicateController(TensorEnd2EndCommunicateController &&) = delete;

    virtual StatusCode handleRequest(SharedRequest request);

    virtual StatusCode sendOrRecv(const TensorEnd2EndCommunicateRequest &request);

    virtual ~TensorEnd2EndCommunicateController() {};

protected:
    std::shared_ptr<CommunicationBackend> backend_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSOREND2ENDCOMMUNICATECONTROLLER_H
