//
// Created by LYL232 on 2021/5/31.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORFLOW_TENSOR_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORFLOW_TENSOR_H

#include "common/CommonTensor.h"
#include "tensorflow/core/framework/tensor.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorflowTensor : public CommonTensor {
public:
    TensorflowTensor(const tensorflow::Tensor &tensor);

    const CommonTensorShape &shape() const;

    size_t byteSize() const noexcept override;

    size_t elements() const noexcept override;

    DataType dtype() const noexcept override;

    void *data() override;

private:
    const tensorflow::Tensor &tensor_;
    mutable CommonTensorShape shape_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORFLOW_TENSOR_H
