//
// Created by LYL232 on 2021/2/10.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSALLREDUCECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSALLREDUCECONTROLLER_H

#include "def.h"
#include "tensorflow/core/framework/tensor.h"


namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce {

/**
 * 按Tensor名字分批次对Tensor进行全规约的控制器
 * 此类本该是抽象类, 但是tensorflow的动态加载库不能够识别抽象类的符号: 错误如下:
 * tensorflow.python.framework.errors_impl.NotFoundError: path-to-lib/lib.so: undefined symbol: _ZN6lyl23210experiment3ddl16tensorsallreduce26TensorsAllreduceControllerD2Ev
 * 遂将其作为一个实体类
 */
class TensorsAllreduceController {
public:
    enum Operation : uchar {
        TENSOR_ALLREDUCE_SUM = 0,
    };

    /**
     * 异步处理Op请求进行Allreduce的Tensor方法: 将Tensor信息提交至后台守护线程后直接返回,
     * 什么时候将哪些Tensor进行规约由具体实现确定
     * @param name 进行规约的Tensor的名字, 要求done方法执行前不能再次提交同样的名字
     * @param sendTensor 进行全规约发送的Tensor
     * @param recvTensor 进行全规约接收的Tensor
     * @param done 异步完成回调函数
     * @return StatusCode 提交到后台线程的状态, 一般为STATUS_OK
     */
    virtual StatusCode handleTenorAllreduceRequest(
            const std::string &name,
            std::shared_ptr<tensorflow::Tensor> sendTensor,
            std::shared_ptr<tensorflow::Tensor> recvTensor,
            std::function<void(StatusCode)> done
    );

    virtual ~TensorsAllreduceController();
};
}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORSALLREDUCECONTROLLER_H
