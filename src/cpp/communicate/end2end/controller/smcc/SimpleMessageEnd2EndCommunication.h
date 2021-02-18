//
// Created by LYL232 on 2021/2/18.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SIMPLEMESSAGEEND2ENDCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SIMPLEMESSAGEEND2ENDCOMMUNICATION_H

#include "def.h"
#include "communicate/end2end/controller/smcc/Message.h"


namespace lyl232 { namespace experiment { namespace ddl { namespace smcc {

class SimpleMessageEnd2EndCommunication {
public:
    typedef std::shared_ptr<Message> SharedMessage;

    virtual StatusCode sendMessage(const Message &message);

    virtual SharedMessage listen();
};

}}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SIMPLEMESSAGEEND2ENDCOMMUNICATION_H
