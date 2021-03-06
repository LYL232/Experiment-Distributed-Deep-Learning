//
// Created by LYL232 on 2021/2/13.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATION_H

#include "def.h"
#include <memory>
#include "communicate/tensor/collective/request/TensorCollectiveCommunicateRequest.h"
#include "Token.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class RingTokenCommunication {
public:
    explicit RingTokenCommunication(std::shared_ptr<Communicator> communicator);

    using Requests = TensorCollectiveCommunicateRequest::Requests;

    virtual void communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const;

    virtual std::shared_ptr<Token> communicationReceiveTokenFrom(int sender) const;

    virtual StatusCode allreduceRequests(const Requests &requests) const;

    virtual StatusCode allgatherRequests(const Requests &requests) const;

    virtual StatusCode broadcastRequests(const Requests &requests) const;

protected:
    std::shared_ptr<Communicator> communicator_;
    static std::shared_ptr<HeapMemoryManager> memManager_;
};


}}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATION_H
