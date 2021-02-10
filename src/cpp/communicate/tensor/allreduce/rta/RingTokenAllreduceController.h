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
#include "Global.h"
#include "communicate/tensor/allreduce/TensorsAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {


#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_TOKEN_DESC_SHOW_MSG 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TF_OP_INTERACTION 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE 1
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE_DETAIL 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_STAGE_CHANGE 0

class Token {

public:
    enum Type : unsigned char {
        TOKEN_TYPE_READY = 0,
        TOKEN_TYPE_SYNC = 1,
        TOKEN_TYPE_ALLREDUCE = 2,
        TOKEN_TYPE_SHUT_DOWN = 3
    };

    Token(Type type, const std::string &msg) : type_(type), msg_(msg) {};

    Token(Type type, std::string &&msg) : type_(type), msg_(msg) {};

    Token(const Token &other) : type_(other.type_), msg_(other.msg_) {}

    Type type() const { return type_; }

    const std::string msg() const { return msg_; }

    std::string &&movingMsg() { return std::move(msg_); }

    const std::string &desc() const;

    /**
     * 判断token是否是停机Token
     * @param token
     * @return bool
     */
    bool isShutDown() { return type_ == TOKEN_TYPE_SHUT_DOWN; }

    static Token shutDownToken() { return Token(TOKEN_TYPE_SHUT_DOWN, "shut down"); }

private:
    Type type_;
    std::string msg_;
    mutable std::string desc_;

};

class TensorEntry {
public:
    TensorEntry(
            const std::string &name,
            const std::shared_ptr<tensorflow::Tensor> &sendTensor,
            const std::shared_ptr<tensorflow::Tensor> &recvTensor,
            std::function<void(StatusCode)> done) :
            name_(name),
            sendTensor_(sendTensor),
            recvTensor_(recvTensor),
            done_(done) {};

    TensorEntry(const TensorEntry &other) = delete;

    const std::string &name() const { return name_; }

    size_t tensorSize() const { return sendTensor_->tensor_data().size(); }

    size_t elements() const { return sendTensor_->shape().num_elements(); }

    tensorflow::DataType dtype() const { return sendTensor_->dtype(); }

    void done(StatusCode status) const { done_(status); }

    void *sendTensorData() const { return (void *) sendTensor_->tensor_data().data(); }

    void *recvTensorData() const { return (void *) recvTensor_->tensor_data().data(); }

private:
    const std::string name_;
    const std::shared_ptr<tensorflow::Tensor> sendTensor_, recvTensor_;
    std::function<void(StatusCode)> done_;
};

class RingTokenAllreduceController: public TensorsAllreduceController {
public:
    RingTokenAllreduceController() :
            outMutex_(PTHREAD_MUTEX_INITIALIZER),
            registerTensorLock_(PTHREAD_RWLOCK_INITIALIZER),
            stageLock_(PTHREAD_RWLOCK_INITIALIZER),
            outputTokenCond_(PTHREAD_COND_INITIALIZER),
            currentStage_(RTAC_INIT),
            sendThread_(&RingTokenAllreduceController::sendMain_, this),
            recvThread_(&RingTokenAllreduceController::recvMain_, this),
            outputtingTokenQueue_(),
            waitingReadyTokenName_(),
            registeredTensors_(),
            sendBuffer_(nullptr), recvBuffer_(nullptr),
            allreduceSendBuffer_(nullptr), allreduceRecvBuffer_(nullptr),
            sendBufferSize_(0), recvBufferSize_(0), allreduceBufferSize_(0),
            tokenMetaSize_(sizeof(Token::Type) + sizeof(size_t)) {}

    RingTokenAllreduceController(const RingTokenAllreduceController &) = delete;

    virtual StatusCode handleTenorAllreduceRequest(
            const std::string &name,
            std::shared_ptr<tensorflow::Tensor> sendTensor,
            std::shared_ptr<tensorflow::Tensor> recvTensor,
            std::function<void(StatusCode)> done
    ) override final;

    bool initialized() const { return currentStage_ != RTAC_INIT; }

    virtual ~RingTokenAllreduceController();

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
    pthread_rwlock_t registerTensorLock_, stageLock_;
    pthread_cond_t outputTokenCond_;
    Stage currentStage_;

    std::thread sendThread_, recvThread_;

    std::queue<std::shared_ptr<Token>> outputtingTokenQueue_;
    std::string waitingReadyTokenName_;
    std::map<std::string, TensorEntry *> registeredTensors_;

    MPI_Status statusBuffer_;

    char *sendBuffer_, *recvBuffer_,
            *allreduceSendBuffer_, *allreduceRecvBuffer_;
    size_t sendBufferSize_, recvBufferSize_,
            allreduceBufferSize_, tokenMetaSize_;

    static double inflateFactor_;
    static std::string tokenNameSplitDelimiter_;

    void fromStageToStage_(Stage from, Stage to);

    void forceToStage_(Stage to);

    bool inStage_(Stage stage);

    void checkSendBuffer_(size_t bytesRequire);

    void checkRecvBuffer_(size_t bytesRequire);

    void checkAllreduceBuffer_(size_t bytesRequire);

    /**
     * 后台发送线程主函数
     */
    void sendMain_();

    void fillTokenSendBufferAndNotify(std::shared_ptr<Token> token);

    /**
     * 后台接收线程主函数
     */
    void recvMain_();

    std::shared_ptr<Token> receiveTokenFromSender();

    void handleReceivingTokenAsTokenGenerator(std::shared_ptr<Token> token);

    void handleReceivingTokenAsTokenReceiver(std::shared_ptr<Token> token);

    void allreduceByNames(const std::set<std::string> &names);

    static std::set<std::string> getNamesFromToken(const Token &token);

    static std::string stageName_(Stage stage);
};

}}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENALLREDUCECONTROLLER_H
