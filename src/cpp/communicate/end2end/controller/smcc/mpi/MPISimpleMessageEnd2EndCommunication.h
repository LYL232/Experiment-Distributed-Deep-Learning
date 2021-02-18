//
// Created by LYL232 on 2021/2/18.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPISIMPLEMESSAGEEND2ENDCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPISIMPLEMESSAGEEND2ENDCOMMUNICATION_H

#include "communicate/end2end/controller/smcc/SimpleMessageEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace smcc {

class MPISimpleMessageEnd2EndCommunication: public SimpleMessageEnd2EndCommunication {
public:
    virtual StatusCode sendMessage(const Message &message) override;

    virtual SharedMessage listen() override;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPISIMPLEMESSAGEEND2ENDCOMMUNICATION_H
