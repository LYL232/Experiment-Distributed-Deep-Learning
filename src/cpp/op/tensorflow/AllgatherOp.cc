//
// Created by LYL232 on 2021/5/29.
//

#include "op/tensorflow/AllgatherOp.h"
#include "global/Global.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/tensor/collective/request/TensorAllgatherRequest.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/tensorflow/TensorflowTensor.h"
#include "common/tensorflow/TensorflowOpContext.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("Allgather")
        .Attr("T: {int32, int64, float32, float64}")
        .Attr("communicator_id: int")
        .Input("tensor: T")
        .Output("allgathered: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

AllgatherOp::AllgatherOp(tensorflow::OpKernelConstruction *context) :
        AsyncOpKernel(context), communicatorId_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("communicator_id", &communicatorId_));
}

void AllgatherOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace lyl232::experiment::ddl;
    using namespace std;
    // 获取输入 tensor
    const Tensor &input = context->input(0);

    auto &global = Global::get();
    auto &controller = global.collectiveCommunicateController();

    OP_REQUIRES_OK_ASYNC(context, statusCode2TFStatus(
            controller.handleRequest(
                    make_shared<TensorAllgatherRequest>(
                            controller,
                            name(),
                            std::make_shared<TensorflowTensor>(input),
                            // allgather并不能在此申请输出张量的内存, 所以放在通信时申请
                            std::shared_ptr<TensorflowTensor>(),
                            [this, context, done](StatusCode code) {
                                context->SetStatus(statusCode2TFStatus(code));
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LOG_OP_DONE_TIME_POINT
                                MS_TIME_LOG("ReceiveTensorOp done:" << name())
#endif
                                done();
                            },
                            global.getCommunicator(communicatorId_),
                            std::make_shared<TensorflowOpContext>(*context)
                    )
            )), done
    );
}

REGISTER_KERNEL_BUILDER(Name("Allgather").Device(DEVICE_CPU), AllgatherOp)
}}}
