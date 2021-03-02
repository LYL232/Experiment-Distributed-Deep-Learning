//
// Created by LYL232 on 2021/2/12.
//

#include "op/BroadcastOp.h"
#include "global/Global.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/tensor/collective/broadcast/TensorBroadcastRequest.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("Broadcast")
        .Attr("T: {int32, int64, float32, float64}")
        .Attr("root_rank: int")
        .Input("tensor: T")
        .Output("broadcasted: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

BroadcastOp::BroadcastOp(tensorflow::OpKernelConstruction *context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &rootRank_));
}

void BroadcastOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace lyl232::experiment::ddl;
    using namespace std;
    // 获取输入 tensor
    const Tensor &input = context->input(0);
    Tensor *output = nullptr;
    // todo: 优化: 让根节点不需要申请内存, 直接使用input作为输出
    OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done
    );

    auto &controller = Global::get().collectiveCommunicateController();

    OP_REQUIRES_OK_ASYNC(context, statusCode2TFStatus(
            controller.handleRequest(
                    make_shared<TensorBroadcastRequest>(
                            controller,
                            name(),
                            std::make_shared<Tensor>(input),
                            std::make_shared<Tensor>(*output),
                            [context, done](StatusCode code) {
                                context->SetStatus(statusCode2TFStatus(code));
                                done();
                            },
                            rootRank_
                    )
            )), done);
}

REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), BroadcastOp)
}}}