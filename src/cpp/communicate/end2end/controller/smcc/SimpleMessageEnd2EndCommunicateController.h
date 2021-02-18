//
// Created by LYL232 on 2021/2/18.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SIMPLEMESSAGEEND2ENDCOMMUNICATECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SIMPLEMESSAGEEND2ENDCOMMUNICATECONTROLLER_H

#include <thread>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include "pthread.h"
#include "communicate/end2end/controller/smcc/Message.h"
#include "communicate/end2end/controller/smcc/SimpleMessageEnd2EndCommunication.h"
#include "communicate/end2end/controller/TensorEnd2EndCommunicateController.h"

#define SMCC_RECEIVING_UNEXPECTED_MESSAGE(msg, rank) { \
    string what("receive unexpected Messgae, sender="); \
    what.append(to_string(msg->sender())).append(", receiver=") \
    .append(to_string(msg->receiver())) \
    .append(", while this rank=") \
    .append(to_string(rank));\
    throw std::runtime_error(what); \
}

namespace lyl232 { namespace experiment { namespace ddl { namespace smcc {

class SimpleMessageEnd2EndCommunicateController : public TensorEnd2EndCommunicateController {
public:
    typedef std::shared_ptr<Message> SharedMessage;
    typedef std::tuple<int, int, std::string> RequestIdentifier;

    struct RequestIdentifierHash : public std::unary_function<RequestIdentifier, std::size_t> {
        std::hash<std::string> stringHash_;

        std::size_t operator()(const RequestIdentifier &k) const {
            return std::get<0>(k) ^ std::get<1>(k) ^ stringHash_(std::get<2>(k));
        }
    };

    SimpleMessageEnd2EndCommunicateController(
            std::shared_ptr<CommunicationBackend> backend,
            std::shared_ptr<SimpleMessageEnd2EndCommunication> communicationImpl
    );

    SimpleMessageEnd2EndCommunicateController(const SimpleMessageEnd2EndCommunicateController &) = delete;

    SimpleMessageEnd2EndCommunicateController(SimpleMessageEnd2EndCommunicateController &&) = delete;

    virtual StatusCode handleRequest(std::shared_ptr<TensorEnd2EndCommunicateRequest> request) override;

    virtual StatusCode sendOrRecv(const TensorEnd2EndCommunicateRequest &request) override;

    virtual ~SimpleMessageEnd2EndCommunicateController();

    bool initialized() const noexcept;

private:
    // 监控表示controller状态的对象成员变量的读写锁, 监控的变量有:
    // senderReady_, receiverReady_, communicatorReady_,
    // registeredRequests_ waitingRequests_, bothReadyRequests_ (其实这行的变量可以另起一个甚至两个读写锁, 但是因为上面三个变量的写要求只存在于初始化, 冲突很少, 所以可以共用)
    mutable pthread_rwlock_t rwLock_;

    std::queue<SharedMessage> messagesToSend_, communicateMessages_;
    pthread_mutex_t messagesToSendMutex_, communicateMutex_;
    pthread_cond_t newMessageToSendCond_, newCommunicateTaskCond_;

    bool senderReady_, receiverReady_, communicatorReady_;

    std::thread sender_, receiver_, communicator_;

    std::shared_ptr<SimpleMessageEnd2EndCommunication> communicationImpl_;

    std::unordered_map<RequestIdentifier, SharedRequest, RequestIdentifierHash> registeredRequests_;

    std::unordered_set<RequestIdentifier, RequestIdentifierHash> waitingRequests_, bothReadyRequests_;

    bool requestReady_(const RequestIdentifier &id) const noexcept;

    void setReady_(bool &readyBool) noexcept;

    void setRequestReady(int sender, int receiver, const std::string &key) noexcept;

    void setRequestReady(int sender, int receiver, std::string &&key) noexcept;

    void messageSenderMain_();

    void messageReceiverMain_();

    void communicatorMain_();

    static void newMessageForwarding_(
            pthread_mutex_t *handlingQueueMutex,
            pthread_cond_t *handlingCond,
            std::queue<SharedMessage> &handlingQueue,
            SharedMessage msg
    );

    static void newMessageForwarding_(
            pthread_mutex_t *handlingQueueMutex,
            pthread_cond_t *handlingCond,
            std::queue<SharedMessage> &handlingQueue,
            std::queue<SharedMessage> &msgs
    );

    void waitForMessages_(
            pthread_mutex_t *queueMutex,
            pthread_cond_t *waitCond,
            bool &readyFlag,
            std::queue<SharedMessage> &queue,
            std::queue<SharedMessage> &result
    );

};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SIMPLEMESSAGEEND2ENDCOMMUNICATECONTROLLER_H
