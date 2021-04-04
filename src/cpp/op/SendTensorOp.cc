//
// Created by LYL232 on 2021/3/1.
//

#include "op/SendTensorOp.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "global/Global.h"
#include "communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.h"


namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("SendTensor")
        .Attr("T: {int32, int64, float32, float64}")
        .Input("input: T")
        .Attr("receiver: int")
        .Attr("communicator_id: int")
        .SetShapeFn(shape_inference::NoOutputs);

SendTensorOp::SendTensorOp(tensorflow::OpKernelConstruction *context) :
        AsyncOpKernel(context), receiver_(-1), communicatorId_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("receiver", &receiver_));
    OP_REQUIRES_OK(context, context->GetAttr("communicator_id", &communicatorId_));
}

void SendTensorOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace std;
    const Tensor &input = context->input(0);

    auto &global = Global::get();

    global.end2EndCommunicateController().handleRequest(
            make_shared<TensorSendCommunicateRequest>(
                    global.end2EndCommunicateController(),
                    name(),
                    std::make_shared<Tensor>(input),
                    [context, done](StatusCode code) {
                        context->SetStatus(statusCode2TFStatus(code));
                        done();
                    },
                    receiver_, global.getCommunicator(communicatorId_)
            )
    );
}

REGISTER_KERNEL_BUILDER(Name("SendTensor").Device(DEVICE_CPU), SendTensorOp)

}}}