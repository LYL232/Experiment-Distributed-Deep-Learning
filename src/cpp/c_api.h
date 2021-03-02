//
// Created by LYL232 on 2021/2/7.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H

#include "communicate/message/message.h"

namespace lyl232 { namespace experiment { namespace ddl {

extern "C" {

int processes();

int process_rank();

Message *listen_message();

void destroy_message(Message *messagePtr);

// example
// str_ptr = c_char_p('abc')
// c_api.snd_message(str_ptr, receiver)
// todo: 返回状态码
void send_message(const char *msg, int receiverRank);

}

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H
