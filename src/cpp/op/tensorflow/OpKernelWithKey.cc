//
// Created by LYL232 on 2021/8/11.
//

#include "op/tensorflow/OpKernelWithKey.h"


namespace lyl232 { namespace experiment { namespace ddl {

OpKernelWithKey::OpKernelWithKey(tensorflow::OpKernelConstruction *context) :
        tensorflow::OpKernel(context), key_() {}

const std::string OpKernelWithKey::key() const {
    if (key_.length() > 0) {
        return key_;
    }
    return name();
}

}}}