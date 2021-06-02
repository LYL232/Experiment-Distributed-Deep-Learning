//
// Created by LYL232 on 2021/5/31.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_OP_CONTEXT_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_OP_CONTEXT_H

#include "def.h"
#include "common/CommonTensor.h"
#include "global/HeapMemoryManager.h"

namespace lyl232 { namespace experiment { namespace ddl {

class OpContext {
public:
    virtual StatusCode allocateOutput(const CommonTensorShape &shape, std::shared_ptr<CommonTensor> &tensor);

    virtual ~OpContext() = default;
protected:
    static std::shared_ptr<HeapMemoryManager> memManager_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_OP_CONTEXT_H
