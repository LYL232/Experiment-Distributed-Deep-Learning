//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENALLREDUCECOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENALLREDUCECOMMUNICATION_H

#include <memory>
#include "def.h"
#include "communicate/tensor/allreduce/rta/Token.h"
#include "communicate/tensor/allreduce/TensorAllreduceRequest.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {


class RingTokenAllreduceCommunication {
public:
    virtual void communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const;

    virtual std::shared_ptr<Token> communicationReceiveTokenFrom(int sender) const;

    virtual StatusCode allreduceRequests(
            const std::map<std::string, TensorAllreduceRequest *> &requests,
            size_t elements, size_t byteSize
    ) const;
};


}}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENALLREDUCECOMMUNICATION_H
