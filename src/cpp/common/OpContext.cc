//
// Created by LYL232 on 2021/5/31.
//

#include "common/OpContext.h"
#include "global/initialize.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::shared_ptr<HeapMemoryManager> OpContext::memManager_(heapMemoryManagerGetter());

StatusCode OpContext::allocateOutput(const CommonTensorShape &shape, std::shared_ptr<CommonTensor> &tensor) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "OpContext::allocateOutput(const CommonTensorShape &shape, std::shared_ptr<CommonTensor> &tensor)");
}

}}}