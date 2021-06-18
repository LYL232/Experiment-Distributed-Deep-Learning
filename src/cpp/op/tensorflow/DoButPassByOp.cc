//
// Created by LYL232 on 2021/6/19.
//

#include "op/tensorflow/DoButPassByOp.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("DoButPassBy")
        .Attr("T: {int32, int64, float32, float64}")
        .Input("forward: T")
        .Input("do: T")
        .Output("forwarded: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

DoButPassByOp::DoButPassByOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {

}

void DoButPassByOp::Compute(tensorflow::OpKernelContext *context) {
    // 仅仅是转发tensor
    const Tensor &forward = context->input(0);
    if (context->input_is_ref(0)) {
        context->forward_ref_input_to_ref_output(0, 0);
    } else {
        context->set_output(0, forward);
    }
}

REGISTER_KERNEL_BUILDER(Name("DoButPassBy").Device(DEVICE_CPU), DoButPassByOp)

}}}