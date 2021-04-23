//
// Created by LYL232 on 2021/2/7.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H

#include "communicate/message/message.h"
#include "communicate/backend/Communicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

extern "C" {

int communicator_rank(Communicator::ID ptr);

int communicator_size(Communicator::ID ptr);

Communicator::ID world_communicator();

Message *listen_message(Communicator::ID communicatorId);

void destroy_message(Message *messagePtr);

// example
// str_ptr = c_char_p('abc')
// c_api.snd_message(str_ptr, receiver, Communicator.id, len)
// todo: 返回状态码
void send_message(const char *msg, int receiverRank, Communicator::ID communicatorId, size_t len);

Message *broadcast_message(const char *msg, int root, Communicator::ID communicatorId, size_t len);

Communicator::ID split_communicator(Communicator::ID communicatorId, int color, int key);

void detach_communicator(Communicator::ID communicatorId);

void py_info(const char *logStr);

void py_debug(const char *logStr);

void py_error(const char *logStr);

}

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H
