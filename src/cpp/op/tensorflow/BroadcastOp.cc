//
// Created by LYL232 on 2021/2/12.
//

#include "op/tensorflow/BroadcastOp.h"
#include "global/Global.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/tensor/collective/request/TensorBroadcastRequest.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/tensorflow/TensorflowTensor.h"
#include "common/tensorflow/TensorflowOpContext.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("Broadcast")
        .Attr("T: numbertype")
        .Attr("communicator_id: int")
        .Attr("root_rank: int")
        .Attr("key: string = ''")
        .Input("tensor: T")
        .Output("broadcasted: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

BroadcastOp::BroadcastOp(tensorflow::OpKernelConstruction *context) :
        AsyncOpKernelWithKey(context), rootRank_(-1), communicatorId_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &rootRank_));
    OP_REQUIRES_OK(context, context->GetAttr("communicator_id", &communicatorId_));
    OP_REQUIRES_OK(context, context->GetAttr("key", &key_));
}

void BroadcastOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace lyl232::experiment::ddl;
    using namespace std;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LOG_OP_BEGIN_TIME_POINT
    MS_TIME_LOG("BroadcastOp begin:" << key())
#endif

    // 获取输入 tensor
    const Tensor &input = context->input(0);
    Tensor *output = nullptr;
    // todo: 优化: 让根节点不需要申请内存, 直接使用input作为输出
    OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done
    );

    auto &global = Global::get();
    auto &controller = global.collectiveCommunicateController();

    OP_REQUIRES_OK_ASYNC(context, statusCode2TFStatus(
            controller.handleRequest(
                    make_shared<TensorBroadcastRequest>(
                            controller,
                            key(),
                            std::make_shared<TensorflowTensor>(input),
                            std::make_shared<TensorflowTensor>(*output),
                            [this, context, done](StatusCode code) {
                                context->SetStatus(statusCode2TFStatus(code));
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LOG_OP_DONE_TIME_POINT
                                MS_TIME_LOG("BroadcastOp done:" << key())
#endif
                                done();
                            },
                            rootRank_, global.getCommunicator(communicatorId_),
                            std::make_shared<TensorflowOpContext>(*context)
                    )
            )), done);
}

REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), BroadcastOp)
}}}