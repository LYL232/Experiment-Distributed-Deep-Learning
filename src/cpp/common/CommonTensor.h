//
// Created by LYL232 on 2021/5/31.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMON_TENSOR_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMON_TENSOR_H

#include <cstddef>
#include <vector>
#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

class CommonTensorShape {
public:
    CommonTensorShape() = default;

    CommonTensorShape(const CommonTensorShape &other);

    CommonTensorShape(CommonTensorShape &&other);

    void addDim(size_t dim);

    void appendShape(const CommonTensorShape &shape);

    size_t dimSize(size_t idx) const;

    size_t numElements() const;

    size_t dims() const;

private:
    std::vector<size_t> shape_;
    mutable size_t numElements_ = 0;
};

class CommonTensor {
public:
    virtual const CommonTensorShape &shape() const;

    virtual size_t byteSize() const;

    virtual size_t elements() const;

    virtual DataType dtype() const;

    virtual void *data();

};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMON_TENSOR_H
