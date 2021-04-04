//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORCOMMUNICATEREQUEST_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORCOMMUNICATEREQUEST_H

#include <memory>
#include "def.h"
#include "communicate/backend/Communicator.h"
#include "tensorflow/core/framework/tensor.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorCommunicateRequest {
public:
    /**
     * 检查两个Tensor的size和dtype是否一致再返回
     * 出现不一致则抛出异常
     * @param key 请求的key, 要求: 一个key只能对应一个未完成的请求(done没有被调用)
     * @param requestingTensor 进行请求的Tensor
     * @param done 完成时回调函数, 通过此函数通知此请求已经完成
     */
    TensorCommunicateRequest(
            std::string key,
            std::shared_ptr<tensorflow::Tensor> requestingTensor,
            std::function<void(StatusCode)> done,
            std::shared_ptr<Communicator> communicator
    );

    TensorCommunicateRequest(const TensorCommunicateRequest &) noexcept = default;

    TensorCommunicateRequest(TensorCommunicateRequest &&) noexcept;

    const std::string &key() const noexcept;

    size_t tensorSize() const noexcept;

    size_t elements() const noexcept;

    tensorflow::DataType dtype() const noexcept;

    std::shared_ptr<tensorflow::Tensor> &requestingTensor() const noexcept;

    void done(StatusCode code) const noexcept;

    void *requestingTensorData() const noexcept;

    const std::shared_ptr<Communicator> &communicator() const noexcept;

    virtual ~TensorCommunicateRequest() = default;

private:
    std::string key_;
    mutable std::shared_ptr<tensorflow::Tensor> requestingTensor_;
    std::function<void(StatusCode)> done_;
    std::shared_ptr<Communicator> communicator_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORCOMMUNICATEREQUEST_H
