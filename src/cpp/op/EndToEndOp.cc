//
// Created by LYL232 on 2021/2/16.
//

#include "op/EndToEndOp.h"
#include "global/Global.h"
#include "communicate/end2end/controller/TensorEnd2EndCommunicateController.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("EndToEnd")
        .Attr("T: {int32, int64, float32, float64}")
        .Attr("sender: int")
        .Attr("receiver: int")
        .Input("tensor: T")
        .Output("communicated: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

EndToEndOp::EndToEndOp(OpKernelConstruction *context) :
        AsyncOpKernel(context), sender_(-1), receiver_(-1) {
    OP_REQUIRES_OK(context, context->GetAttr("sender", &sender_));
    OP_REQUIRES_OK(context, context->GetAttr("receiver", &receiver_));
}

void EndToEndOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace lyl232::experiment::ddl;
    using namespace std;
    // 获取输入 tensor
    const Tensor &input = context->input(0);
    int rank = Global::get().processRank();
    Tensor *output = nullptr;
    if (rank == sender_) {
        // sender不需要输出tensor
        output = &const_cast<Tensor &>(input);
    } else if (rank == receiver_) {
        // 创建输出 tensor, context->allocate_output 用来分配输出内存
        OP_REQUIRES_OK_ASYNC(
                context, context->allocate_output(0, input.shape(), &output), done
        );
    }

    auto &controller = Global::get().end2EndCommunicateController();

    OP_REQUIRES_OK_ASYNC(context, statusCode2TFStatus(
            controller.handleRequest(
                    make_shared<TensorEnd2EndCommunicateRequest>(
                            controller,
                            name(),
                            std::make_shared<Tensor>(input),
                            std::make_shared<Tensor>(*output),
                            [context, done](StatusCode code) {
                                context->SetStatus(statusCode2TFStatus(code));
                                done();
                            },
                            sender_, receiver_
                    )
            )), done);
}

REGISTER_KERNEL_BUILDER(Name("EndToEnd").Device(DEVICE_CPU), EndToEndOp)

}}}