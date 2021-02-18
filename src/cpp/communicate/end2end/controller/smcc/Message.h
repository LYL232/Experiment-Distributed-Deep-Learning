//
// Created by LYL232 on 2021/2/18.
//

#ifndef EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SMCC_MESSAGE_H
#define EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SMCC_MESSAGE_H

#include <string>

namespace lyl232 { namespace experiment { namespace ddl { namespace smcc {

class Message {
public:
    enum Type : unsigned char {
        READY,
        COMMUNICATE,
        DONE
    };

    // todo: 进行检查: sender != receiver
    Message(Type type, const std::string &msg, int sender, int receiver);

    Message(Type type, std::string &&msg, int sender, int receiver);

    Message(const Message &other);

    Message(Message &&other);

    Type type() const noexcept;

    const std::string &message() const noexcept;

    int sender() const noexcept;

    int receiver() const noexcept;

    std::string &&movingMessage() noexcept;

    const std::string &desc() const noexcept;

private:
    Type type_;
    std::string msg_;
    mutable std::string desc_;
    int sender_, receiver_;
};

}}}}

#endif //EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SMCC_MESSAGE_H
