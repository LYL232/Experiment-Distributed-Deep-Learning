//
// Created by LYL232 on 2021/8/8.
//

#include "op/tensorflow/TimeLogOp.h"
#include "global/Global.h"
#include "communicate/message/MessageController.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"
#include "tensorflow/core/framework/shape_inference.h"


namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("TimeLog")
        .Attr("T: {int32, int64, float32, float64}")
        .Input("pass: T")
        .Output("passed: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

TimeLogOp::TimeLogOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}

void TimeLogOp::Compute(OpKernelContext *context) {
    using namespace std;
    const Tensor &sendingTensor = context->input(0);
    // 仅仅是转发tensor
    if (context->input_is_ref(0)) {
        context->forward_ref_input_to_ref_output(0, 0);
    } else {
        context->set_output(0, sendingTensor);
    }
    MS_TIME_LOG("[TIME-LOG]: " << name())
}

REGISTER_KERNEL_BUILDER(Name("TimeLog").Device(DEVICE_CPU), TimeLogOp)

}}}

