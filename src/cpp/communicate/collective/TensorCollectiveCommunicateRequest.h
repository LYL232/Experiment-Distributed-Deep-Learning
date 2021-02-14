//
// Created by LYL232 on 2021/2/10.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORCOLLECTIVECOMMUNICATEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORCOLLECTIVECOMMUNICATEREQUEST_H

#include <memory>
#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorsCollectiveCommunicateController;

/**
 * OP要求对Tensor进行组通信时的请求类, 提供一系列访问请求Tensor和结果Tensor的方法, 还有维护计算完毕时
 * 回调的异步方法
 */
class TensorCollectiveCommunicateRequest {
public:
    typedef std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>> Requests;

    /**
     * 检查两个Tensor的size和dtype是否一致再返回
     * 出现不一致则抛出异常
     * @param key 请求的key, 要求: 一个key只能对应一个未完成的请求(done没有被调用)
     * @param requestingTensor 进行请求的Tensor
     * @param resultTensor 该请求返回结果的Tensor
     * @param done 完成时回调函数, 通过此函数通知此请求已经完成
     */
    TensorCollectiveCommunicateRequest(
            TensorsCollectiveCommunicateController &controller,
            const std::string &key,
            std::shared_ptr<tensorflow::Tensor> requestingTensor,
            std::shared_ptr<tensorflow::Tensor> resultTensor,
            std::function<void(StatusCode)> done
    );

    TensorCollectiveCommunicateRequest(const TensorCollectiveCommunicateRequest &) noexcept;

    TensorCollectiveCommunicateRequest(TensorCollectiveCommunicateRequest &&) noexcept;

    const std::string &key() const noexcept;

    size_t tensorSize() const noexcept;

    size_t elements() const noexcept;

    tensorflow::DataType dtype() const noexcept;

    std::shared_ptr<tensorflow::Tensor> &requestingTensor() const noexcept;

    std::shared_ptr<tensorflow::Tensor> &resultTensor() const noexcept;

    void done(StatusCode code) const noexcept;

    void *requestingTensorData() const noexcept;

    void *resultTensorData() const noexcept;

    virtual StatusCode doCollectiveCommunication(const Requests &requests);

    virtual const char *requestTypeName() const;

    virtual ~TensorCollectiveCommunicateRequest() {};
protected:
    TensorsCollectiveCommunicateController &controller_;
private:
    std::string key_;
    mutable std::shared_ptr<tensorflow::Tensor> requestingTensor_, resultTensor_;
    std::function<void(StatusCode)> done_;


    static void checkTensorSize_(
            const std::shared_ptr<tensorflow::Tensor> &requestingTensor,
            const std::shared_ptr<tensorflow::Tensor> &resultTensor
    );
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORCOLLECTIVECOMMUNICATEREQUEST_H
