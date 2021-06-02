//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGECONTROLLER_H

#include <memory>
#include "communicate/message/Message.h"
#include "communicate/backend/Communicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MessageController {
public:
    virtual Message *broadcastMessage(
            const Message &message, int root, const std::shared_ptr<Communicator> &communicator);

    virtual void sendMessage(const Message &message, int receiver, const std::shared_ptr<Communicator> &communicator);

    virtual Message *listen(const Communicator &communicator);

protected:
    static std::shared_ptr<HeapMemoryManager> memManager_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGECONTROLLER_H
