//
// Created by LYL232 on 2021/8/11.
//

#include "op/tensorflow/AsyncOpKernelWithKey.h"

namespace lyl232 { namespace experiment { namespace ddl {

AsyncOpKernelWithKey::AsyncOpKernelWithKey(tensorflow::OpKernelConstruction *context) :
        tensorflow::AsyncOpKernel(context), key_() {}

const std::string AsyncOpKernelWithKey::key() const {
    if (key_.length() > 0) {
        return key_;
    }
    return name();
}
}}}