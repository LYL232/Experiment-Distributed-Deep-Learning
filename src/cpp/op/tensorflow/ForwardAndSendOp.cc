//
// Created by LYL232 on 2021/2/27.
//

#include "op/tensorflow/ForwardAndSendOp.h"
#include "global/Global.h"
#include "communicate/message/MessageController.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/tensorflow/TensorflowTensor.h"
#include "common/tensorflow/TensorflowOpContext.h"


namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("ForwardAndSend")
        .Attr("T: {int32, int64, float32, float64}")
        .Input("forward: T")
        .Input("send: T")
        .Attr("receiver: int")
        .Attr("communicator_id: int")
        .Attr("tag: int")
        .Output("forwarded: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

ForwardAndSendOp::ForwardAndSendOp(tensorflow::OpKernelConstruction *context) :
        AsyncOpKernel(context), receiver_(-1), tag_(-1), communicatorId_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("receiver", &receiver_));
    OP_REQUIRES_OK(context, context->GetAttr("tag", &tag_));
    OP_REQUIRES_OK(context, context->GetAttr("communicator_id", &communicatorId_));
}

void ForwardAndSendOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace std;
    const Tensor &forward = context->input(0);
    const Tensor &sendingTensor = context->input(1);

    // 仅仅是转发tensor
    if (context->input_is_ref(0)) {
        context->forward_ref_input_to_ref_output(0, 0);
    } else {
        context->set_output(0, forward);
    }

    auto &global = Global::get();
    const auto &commPtr = global.getCommunicator(communicatorId_);

    global.end2EndCommunicateController().handleRequest(
            make_shared<TensorSendCommunicateRequest>(
                    global.end2EndCommunicateController(),
                    name(),
                    std::make_shared<TensorflowTensor>(sendingTensor),
                    [context, done](StatusCode code) {
                        context->SetStatus(statusCode2TFStatus(code));
                        done();
                    },
                    receiver_, commPtr,
                    std::make_shared<TensorflowOpContext>(*context),
                    tag_
            )
    );

}

REGISTER_KERNEL_BUILDER(Name("ForwardAndSend").Device(DEVICE_CPU), ForwardAndSendOp)

}}}

