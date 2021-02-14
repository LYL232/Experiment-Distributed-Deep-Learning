//
// Created by LYL232 on 2021/2/13.
//

#include "Token.h"
#include "communicate/collective/allreduce/TensorAllreduceRequest.h"
#include "communicate/collective/broadcast/TensorBroadcastRequest.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

const char *Token::shutdownTypeName_ = "ShutDown";

std::pair<std::string, Token::RequestType> Token::requestNameMapInitilizer_[] = {
        std::make_pair(Token::shutdownTypeName_, Token::TOKEN_REQUEST_SHUTDOWN),
        std::make_pair(TensorBroadcastRequest::requestType, Token::TOKEN_REQUEST_BROADCAST),
        std::make_pair(TensorAllreduceRequest::requestType, Token::TOKEN_REQUEST_ALLREDUCE)
};

std::map<std::string, Token::RequestType> Token::requestNameMap_(
        requestNameMapInitilizer_,
        requestNameMapInitilizer_ +
        (sizeof(requestNameMapInitilizer_) / sizeof(requestNameMapInitilizer_[0]))
);

const std::string &Token::desc() const {
    using namespace std;
    if (desc_.length() == 0) {
        desc_.append("{type: ");
        switch (type_) {
            case TOKEN_TYPE_READY:
                desc_.append("TOKEN_TYPE_READY");
                break;
            case TOKEN_TYPE_SYNC:
                desc_.append("TOKEN_TYPE_SYNC");
                break;
            case TOKEN_TYPE_COMMUNICATE:
                desc_.append("TOKEN_TYPE_COMMUNICATE");
                break;
            case TOKEN_TYPE_SHUT_DOWN:
                desc_.append("TOKEN_TYPE_SHUT_DOWN");
                break;
        }
        desc_.append(", requestType: ");
        switch (requestType_) {
            case TOKEN_REQUEST_SHUTDOWN:
                desc_.append("TOKEN_REQUEST_SHUTDOWN");
                break;
            case TOKEN_REQUEST_ALLREDUCE:
                desc_.append("TOKEN_REQUEST_ALLREDUCE");
                break;
            case TOKEN_REQUEST_BROADCAST:
                desc_.append("TOKEN_REQUEST_BROADCAST");
                break;
            case TOKEN_REQUEST_UNKNOWN:
                desc_.append("TOKEN_REQUEST_UNKNOWN");
                break;
        }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_TOKEN_DESC_SHOW_MSG
        desc_.append(", msg: ");
        if (msg_.find("\n") != string::npos) {
            desc_.append("\n");
        }
        desc_.append(msg_);
#endif
        desc_.append("}");
    }
    return desc_;
}

Token::Token(Token::Type type, Token::RequestType requestType, const std::string &msg) :
        type_(type), requestType_(requestType), msg_(msg) {}

Token::Token(Token::Type type, Token::RequestType requestType, std::string &&msg) :
        type_(type), requestType_(requestType), msg_(std::move(msg)) {}

Token::Token(Token &&other) :
        type_(other.type_), requestType_(other.requestType_),
        msg_(std::move(other.msg_)) {}

Token::Type Token::type() const {
    return type_;
}

Token::RequestType Token::requestType() const {
    return requestType_;
}

const std::string Token::msg() const {
    return msg_;
}

std::string &&Token::movingMsg() {
    return std::move(msg_);
}

const char *Token::requestTypeName() const {
    return requestTypeName(requestType_);
}

Token::RequestType Token::requestType(const std::string &requestTypeName) {
    auto iter = requestNameMap_.find(requestTypeName);
    if (iter == requestNameMap_.end()) {
        return TOKEN_REQUEST_UNKNOWN;
    }
    return iter->second;
}

const char *Token::requestTypeName(Token::RequestType type) {
    switch (type) {
        case TOKEN_REQUEST_SHUTDOWN:
            return shutdownTypeName_;
        case TOKEN_REQUEST_ALLREDUCE:
            return TensorAllreduceRequest::requestType;
        case TOKEN_REQUEST_BROADCAST:
            return TensorBroadcastRequest::requestType;
        default:
            return "ERROR: Unknown Token::RequestType";
    }
}

}}}}