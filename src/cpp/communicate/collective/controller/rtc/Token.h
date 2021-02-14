//
// Created by LYL232 on 2021/2/13.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TOKEN_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TOKEN_H

#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_TOKEN_DESC_SHOW_MSG 0

#include <string>
#include <map>

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class Token {
public:
    enum Type : unsigned char {
        TOKEN_TYPE_READY = 0,
        TOKEN_TYPE_SYNC = 1,
        TOKEN_TYPE_COMMUNICATE = 2,
        TOKEN_TYPE_SHUT_DOWN = 3
    };

    enum RequestType : unsigned char {
        TOKEN_REQUEST_SHUTDOWN = 0,
        TOKEN_REQUEST_ALLREDUCE = 1,
        TOKEN_REQUEST_BROADCAST = 2,
        TOKEN_REQUEST_UNKNOWN
    };

    Token(Type type, RequestType requestType, const std::string &msg);

    Token(Type type, RequestType requestType, std::string &&msg);

    Token(const Token &) = delete;

    Token(Token &&other);

    Type type() const;

    RequestType requestType() const;

    const char *requestTypeName() const;

    const std::string msg() const;

    std::string &&movingMsg();

    const std::string &desc() const;

    /**
     * 判断token是否是停机Token
     * @param token
     * @return bool
     */
    bool isShutDown() { return type_ == TOKEN_TYPE_SHUT_DOWN; }


    static RequestType requestType(const std::string &requestTypeName);

    static const char *requestTypeName(RequestType type);

private:
    Type type_;
    RequestType requestType_;
    std::string msg_;
    mutable std::string desc_;
    static const char *shutdownTypeName_;
    static std::pair<std::string, Token::RequestType> requestNameMapInitilizer_[];
    static std::map<std::string, Token::RequestType> requestNameMap_;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TOKEN_H