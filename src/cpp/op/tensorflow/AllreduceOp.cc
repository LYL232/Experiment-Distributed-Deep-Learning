//
// Created by LYL232 on 2021/2/6.
//

#include "global/Global.h"
#include "op/tensorflow/AllreduceOp.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/tensor/collective/request/TensorAllreduceRequest.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/tensorflow/TensorflowTensor.h"
#include "common/tensorflow/TensorflowOpContext.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("Allreduce")
        .Attr("T: numbertype")
        .Attr("communicator_id: int")
        .Attr("key: string = ''")
        .Input("tensor: T")
        .Output("allreduced: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

AllreduceOp::AllreduceOp(tensorflow::OpKernelConstruction *context) :
        AsyncOpKernelWithKey(context), communicatorId_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("communicator_id", &communicatorId_));
    OP_REQUIRES_OK(context, context->GetAttr("key", &key_));
}

void AllreduceOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace lyl232::experiment::ddl;
    using namespace std;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LOG_OP_BEGIN_TIME_POINT
    MS_TIME_LOG("AllreduceOp begin:" << key())
#endif
    // 获取输入 tensor
    const Tensor &input = context->input(0);
    // 创建输出 tensor, context->allocate_output 用来分配输出内存
    Tensor *output = nullptr;
    OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done
    );

    auto &global = Global::get();
    auto &controller = global.collectiveCommunicateController();

    OP_REQUIRES_OK_ASYNC(context, statusCode2TFStatus(
            controller.handleRequest(
                    make_shared<TensorAllreduceRequest>(
                            controller,
                            key(),
                            std::make_shared<TensorflowTensor>(input),
                            std::make_shared<TensorflowTensor>(*output),
                            [this, context, done](StatusCode code) {
                                context->SetStatus(statusCode2TFStatus(code));
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LOG_OP_DONE_TIME_POINT
                                MS_TIME_LOG("AllreduceOp done:" << key())
#endif
                                done();
                            },
                            TensorAllreduceRequest::Operation::ALLREDUCE_OP_SUM,
                            global.getCommunicator(communicatorId_),
                            std::make_shared<TensorflowOpContext>(*context)
                    )
            )), done
    );
}

REGISTER_KERNEL_BUILDER(Name("Allreduce").Device(DEVICE_CPU), AllreduceOp)
}}}
