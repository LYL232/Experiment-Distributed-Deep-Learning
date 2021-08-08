//
// Created by LYL232 on 2021/2/28.
//

#include "tensorflow/core/framework/shape_inference.h"
#include "op/tensorflow/ReceiveTensorOp.h"
#include "global/Global.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"
#include "common/tensorflow/TensorflowTensor.h"
#include "common/tensorflow/TensorflowOpContext.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("ReceiveTensor")
        .Attr("T: {int32, int64, float32, float64}")
        .Input("input: T")
        .Attr("sender: int")
        .Attr("communicator_id: int")
        .Attr("tag: int")
        .Output("received: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

ReceiveTensorOp::ReceiveTensorOp(tensorflow::OpKernelConstruction *context) :
        AsyncOpKernel(context), sender_(-1), tag_(-1), communicatorId_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("sender", &sender_));
    OP_REQUIRES_OK(context, context->GetAttr("tag", &tag_));
    OP_REQUIRES_OK(context, context->GetAttr("communicator_id", &communicatorId_));
}

void ReceiveTensorOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace std;
    const Tensor &input = context->input(0);

    auto &global = Global::get();

    // 将input转换为output
    if (context->input_is_ref(0)) {
        context->forward_ref_input_to_ref_output(0, 0);
    } else {
        context->set_output(0, input);
    }

    global.end2EndCommunicateController().handleRequest(
            make_shared<TensorReceiveCommunicateRequest>(
                    global.end2EndCommunicateController(),
                    name(),
                    make_shared<TensorflowTensor>(input),
                    [this, context, done](StatusCode code) {
                        context->SetStatus(statusCode2TFStatus(code));
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LOG_OP_DONE_TIME_POINT
                        MS_TIME_LOG("ReceiveTensorOp done:" << name())
#endif
                        done();
                    },
                    sender_, global.getCommunicator(communicatorId_),
                    std::make_shared<TensorflowOpContext>(*context),
                    tag_
            )
    );
}

REGISTER_KERNEL_BUILDER(Name("ReceiveTensor").Device(DEVICE_CPU), ReceiveTensorOp)


}}}