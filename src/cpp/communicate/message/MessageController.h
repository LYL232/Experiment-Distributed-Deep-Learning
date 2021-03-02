//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGECONTROLLER_H

#include <memory>
#include "communicate/message/message.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MessageController {
public:
    virtual void sendMessage(const Message &message, int receiver);

    virtual Message *listen();
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGECONTROLLER_H
