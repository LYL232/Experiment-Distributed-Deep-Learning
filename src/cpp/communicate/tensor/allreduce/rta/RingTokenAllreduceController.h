//
// Created by LYL232 on 2021/2/8.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENALLREDUCECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENALLREDUCECONTROLLER_H

#include <thread>
#include <pthread.h>
#include <queue>
#include <map>
#include <set>
#include "tensorflow/core/framework/tensor.h"
#include "global/Global.h"
#include "communicate/communication/CommunicationBackend.h"
#include "communicate/tensor/allreduce/TensorsAllreduceController.h"
#include "communicate/tensor/allreduce/TensorAllreduceRequest.h"
#include "communicate/tensor/allreduce/rta/Token.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {


#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TF_OP_INTERACTION 1
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE 1
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_STAGE_CHANGE 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE_DETAIL 0

class RingTokenAllreduceController : public TensorsAllreduceController {
public:

    RingTokenAllreduceController(
            std::ostream &initializingLogStream,
            std::shared_ptr<CommunicationBackend> communication
    );

    RingTokenAllreduceController(const RingTokenAllreduceController &) = delete;

    RingTokenAllreduceController(RingTokenAllreduceController &&) = delete;

    virtual StatusCode handleTenorAllreduceRequest(
            const std::string &key,
            std::shared_ptr<tensorflow::Tensor> sendTensor,
            std::shared_ptr<tensorflow::Tensor> recvTensor,
            std::function<void(StatusCode)> done,
            Operation op
    ) override final;

    virtual bool initialized() const noexcept override final { return currentStage_ != RTAC_INIT; }

    virtual ~RingTokenAllreduceController();

protected:
    virtual void communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const;

    virtual std::shared_ptr<Token> communicationReceiveTokenFrom(int sender) const;

    virtual StatusCode allreduceRequests(
            const std::map<std::string, TensorAllreduceRequest *> &requests,
            size_t elements, size_t byteSize
    ) const;

    void notifyAndWaitThreadToShutDown();

private:
    enum Stage : int {
        RTAC_INIT = 0,
        RTAC_WAITING_TENSORS = 1,
        RTAC_WAITING_READY_TOKEN = 2,
        RTAC_WAITING_SYNC_TOKEN = 3,
        RTAC_WAITING_ALLREDUCE_TOKEN = 4,
        RTAC_ALLREDUCING = 5,
        RTAC_SHUT_DOWN,
    };

    pthread_mutex_t outMutex_;
    pthread_rwlock_t registerLock_, stageLock_;
    pthread_cond_t outputTokenCond_;
    Stage currentStage_;

    std::thread sendThread_, recvThread_;

    std::queue<std::shared_ptr<Token>> outputtingTokenQueue_;
    std::string waitingReadyTokenName_;
    std::map<std::string, TensorAllreduceRequest *> registeredRequest_;

    static std::string tokenNameSplitDelimiter_;

    void fromStageToStage_(Stage from, Stage to);

    void forceToStage_(Stage to);

    bool inStage_(Stage stage);

    /**
     * 后台发送线程主函数
     */
    void sendMain_(std::shared_ptr<CommunicationBackend> communication);

    void fillTokenSendBufferAndNotify(std::shared_ptr<Token> token);

    /**
     * 后台接收线程主函数
     */
    void recvMain_(std::shared_ptr<CommunicationBackend> communication);

    void handleReceivingTokenAsTokenGenerator(std::shared_ptr<Token> token);

    void handleReceivingTokenAsTokenReceiver(std::shared_ptr<Token> token);

    StatusCode allreduceByNames(const std::set<std::string> &names);

    static std::set<std::string> getNamesFromToken(const Token &token);

    static std::string stageName_(Stage stage);
};

}}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENALLREDUCECONTROLLER_H
