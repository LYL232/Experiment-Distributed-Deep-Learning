//
// Created by LYL232 on 2021/6/19.
//

#include "op/tensorflow/PassWithComputed.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("PassWithComputed")
        .Input("to_pass: T")
        .Input("to_do_ops: N * T")
        .Output("passed: T")
        .Attr("N: int >= 1")
        .Attr("T: numbertype")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

PassWithComputed::PassWithComputed(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}

void PassWithComputed::Compute(tensorflow::OpKernelContext *context) {
    // 仅仅是转发tensor
    const Tensor &forward = context->input(0);
    if (context->input_is_ref(0)) {
        context->forward_ref_input_to_ref_output(0, 0);
    } else {
        context->set_output(0, forward);
    }
}

REGISTER_KERNEL_BUILDER(Name("PassWithComputed").Device(DEVICE_CPU), PassWithComputed)

}}}