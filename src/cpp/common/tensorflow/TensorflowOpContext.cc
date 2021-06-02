//
// Created by LYL232 on 2021/5/31.
//

#include "common/tensorflow/TensorflowOpContext.h"
#include "common/tensorflow/TensorflowTensor.h"
#include "global/Global.h"

namespace lyl232 { namespace experiment { namespace ddl {

TensorflowOpContext::TensorflowOpContext(tensorflow::OpKernelContext &context) : context_(context) {}

StatusCode TensorflowOpContext::allocateOutput(const CommonTensorShape &shape, std::shared_ptr<CommonTensor> &tensor) {
    tensorflow::TensorShape tfShape;
    for (size_t i = 0; i < shape.dims(); ++i) {
        tfShape.AddDim(shape.dimSize(i));
    }
    tensorflow::Tensor *tfTensor;
    tensorflow::Status status = context_.allocate_output(0, tfShape, &tfTensor);
    if (status.ok()) {
        tensor.reset(TRACK_TYPE_ALLOCATE(memManager_, new TensorflowTensor(*tfTensor), TensorflowTensor));
        return STATUS_OK;
    }
    // todo: status code specify
    return STATUS_ERROR_UNKNOWN;
}

}}}