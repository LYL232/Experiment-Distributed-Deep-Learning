//
// Created by LYL232 on 2021/2/6.
//

#include "global/Global.h"
#include "op/AllreduceOp.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/tensor/collective/allreduce/TensorAllreduceRequest.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace lyl232 { namespace experiment { namespace ddl {

using namespace tensorflow;

REGISTER_OP("Allreduce")
        .Attr("T: {int32, int64, float32, float64}")
        .Input("tensor: T")
        .Output("allreduced: T")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return tensorflow::Status::OK();
        });

void AllreduceOp::ComputeAsync(OpKernelContext *context, DoneCallback done) {
    using namespace lyl232::experiment::ddl;
    using namespace std;
    // 获取输入 tensor
    const Tensor &input = context->input(0);
    // 创建输出 tensor, context->allocate_output 用来分配输出内存
    Tensor *output = nullptr;
    OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done
    );

    auto &controller = Global::get().collectiveCommunicateController();

    OP_REQUIRES_OK_ASYNC(context, statusCode2TFStatus(
            controller.handleRequest(
                    make_shared<TensorAllreduceRequest>(
                            controller,
                            name(),
                            std::make_shared<Tensor>(input),
                            std::make_shared<Tensor>(*output),
                            [context, done](StatusCode code) {
                                context->SetStatus(statusCode2TFStatus(code));
                                done();
                            },
                            TensorAllreduceRequest::Operation::ALLREDUCE_OP_SUM
                    )
            )), done
    );
}

REGISTER_KERNEL_BUILDER(Name("Allreduce").Device(DEVICE_CPU), AllreduceOp)
}}}
