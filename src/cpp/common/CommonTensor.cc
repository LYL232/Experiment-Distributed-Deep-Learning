//
// Created by LYL232 on 2021/5/31.
//

#include "common/CommonTensor.h"

namespace lyl232 { namespace experiment { namespace ddl {

CommonTensorShape::CommonTensorShape(const CommonTensorShape &other) :
        shape_(other.shape_), numElements_(other.numElements_) {}

CommonTensorShape::CommonTensorShape(CommonTensorShape &&other) :
        shape_(std::move(other.shape_)), numElements_(other.numElements_) {}

void CommonTensorShape::addDim(size_t dim) {
    assert(dim > 0);
    shape_.emplace_back(dim);
    numElements_ = 0;
}

void CommonTensorShape::appendShape(const CommonTensorShape &shape) {
    for (auto dim : shape.shape_) {
        shape_.emplace_back(dim);
    }
    numElements_ = 0;
}

size_t CommonTensorShape::dimSize(size_t idx) const {
    assert(idx < shape_.size());
    return shape_[idx];
}

size_t CommonTensorShape::numElements() const {
    if (shape_.size() == 0) {
        numElements_ = 1;
        return 1;
    }
    if (numElements_ > 0) {
        return numElements_;
    }
    numElements_ = 1;
    for (size_t dim : shape_) {
        numElements_ *= dim;
    }
    return numElements_;
}

size_t CommonTensorShape::dims() const {
    return shape_.size();
}

const CommonTensorShape &CommonTensor::shape() const {
    CALLING_ABSTRACT_INTERFACE_ERROR("CommonTensor::shape()");
}

size_t CommonTensor::byteSize() const {
    CALLING_ABSTRACT_INTERFACE_ERROR("CommonTensor::byteSize()");
}

size_t CommonTensor::elements() const {
    CALLING_ABSTRACT_INTERFACE_ERROR("CommonTensor::elements()");
}

DataType CommonTensor::dtype() const {
    CALLING_ABSTRACT_INTERFACE_ERROR("CommonTensor::dtype()");
}

void *CommonTensor::data() {
    CALLING_ABSTRACT_INTERFACE_ERROR("CommonTensor::data()");
}
}}}