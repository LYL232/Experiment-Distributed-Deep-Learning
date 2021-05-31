//
// Created by LYL232 on 2021/5/31.
//

#include "common/tensorflow/TensorflowTensor.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorflowTensor::TensorflowTensor(const tensorflow::Tensor &tensor) : tensor_(tensor), shape_() {
    const auto &shape = tensor.shape();
    for (int i = 0; i < shape.dims(); ++i) {
        shape_.addDim(shape.dim_size(i));
    }
}

const CommonTensorShape &TensorflowTensor::shape() const {
    return shape_;
}

size_t TensorflowTensor::byteSize() const noexcept {
    return tensor_.tensor_data().size();
}

size_t TensorflowTensor::elements() const noexcept {
    return tensor_.NumElements();
}

DataType TensorflowTensor::dtype() const noexcept {
    return tensor_.dtype();
}

void *TensorflowTensor::data() {
    return (void *) tensor_.tensor_data().data();
}
}}}