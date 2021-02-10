//
// Created by LYL232 on 2021/2/10.
//

#include "Token.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {

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
            case TOKEN_TYPE_ALLREDUCE:
                desc_.append("TOKEN_TYPE_ALLREDUCE");
                break;
            case TOKEN_TYPE_SHUT_DOWN:
                desc_.append("TOKEN_TYPE_SHUT_DOWN");
                break;
        }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_TOKEN_DESC_SHOW_MSG
        desc_.append(", msg: ").append(msg_);
#endif
        desc_.append("}");
    }
    return desc_;
}

}}}}}