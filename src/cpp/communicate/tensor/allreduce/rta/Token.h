//
// Created by LYL232 on 2021/2/10.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RTA_TOKEN_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RTA_TOKEN_H
#include <string>

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {

#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_TOKEN_DESC_SHOW_MSG 0

class Token {

public:
    enum Type : unsigned char {
        TOKEN_TYPE_READY = 0,
        TOKEN_TYPE_SYNC = 1,
        TOKEN_TYPE_ALLREDUCE = 2,
        TOKEN_TYPE_SHUT_DOWN = 3
    };

    Token(Type type, const std::string &msg) : type_(type), msg_(msg) {};

    Token(Type type, std::string &&msg) : type_(type), msg_(msg) {};

    Token(const Token &other) : type_(other.type_), msg_(other.msg_) {}

    Type type() const { return type_; }

    const std::string msg() const { return msg_; }

    std::string &&movingMsg() { return std::move(msg_); }

    const std::string &desc() const;

    /**
     * 判断token是否是停机Token
     * @param token
     * @return bool
     */
    bool isShutDown() { return type_ == TOKEN_TYPE_SHUT_DOWN; }

    static Token shutDownToken() { return Token(TOKEN_TYPE_SHUT_DOWN, "shut down"); }

private:
    Type type_;
    std::string msg_;
    mutable std::string desc_;

};

}}}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RTA_TOKEN_H
