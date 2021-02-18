//
// Created by LYL232 on 2021/2/18.
//

#include "communicate/end2end/controller/smcc/SimpleMessageEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace smcc {


SimpleMessageEnd2EndCommunicateController::SimpleMessageEnd2EndCommunicateController(
        std::shared_ptr<CommunicationBackend> backend,
        std::shared_ptr<SimpleMessageEnd2EndCommunication> communicationImpl
) :
        TensorEnd2EndCommunicateController(backend),
        rwLock_(PTHREAD_RWLOCK_INITIALIZER),
        messagesToSend_(), communicateMessages_(),
        messagesToSendMutex_(PTHREAD_MUTEX_INITIALIZER), communicateMutex_(PTHREAD_MUTEX_INITIALIZER),
        newMessageToSendCond_(PTHREAD_COND_INITIALIZER), newCommunicateTaskCond_(PTHREAD_COND_INITIALIZER),
        senderReady_(false), receiverReady_(false), communicatorReady_(false),
        sender_(&SimpleMessageEnd2EndCommunicateController::messageSenderMain_, this),
        receiver_(&SimpleMessageEnd2EndCommunicateController::messageReceiverMain_, this),
        communicator_(&SimpleMessageEnd2EndCommunicateController::communicatorMain_, this),
        communicationImpl_(communicationImpl),
        registeredRequests_(), waitingRequests_(), bothReadyRequests_() {
    while (!initialized()) {
        // 忙等待所有线程准备完毕
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

StatusCode
SimpleMessageEnd2EndCommunicateController::handleRequest(SharedRequest request) {
    return TensorEnd2EndCommunicateController::handleRequest(request);
}

bool SimpleMessageEnd2EndCommunicateController::initialized() const noexcept {
    pthread_rwlock_rdlock(&rwLock_);
    bool res = senderReady_ && receiverReady_ && communicatorReady_;
    pthread_rwlock_unlock(&rwLock_);
    return res;
}

bool SimpleMessageEnd2EndCommunicateController::requestReady_(
        const RequestIdentifier &id) const noexcept {
    pthread_rwlock_rdlock(&rwLock_);
    bool res = registeredRequests_.find(id) != registeredRequests_.end();
    pthread_rwlock_unlock(&rwLock_);
    return res;
}


void SimpleMessageEnd2EndCommunicateController::setReady_(bool &readyBool) noexcept {
    pthread_rwlock_rdlock(&rwLock_);
    if (!readyBool) {
        pthread_rwlock_unlock(&rwLock_);
        pthread_rwlock_wrlock(&rwLock_);
        readyBool = true;
    }
    pthread_rwlock_unlock(&rwLock_);
}

void SimpleMessageEnd2EndCommunicateController::setRequestReady(
        int sender, int receiver, const std::string &key) noexcept {
    pthread_rwlock_wrlock(&rwLock_);
    bothReadyRequests_.emplace(sender, receiver, key);
    pthread_rwlock_unlock(&rwLock_);
}

void SimpleMessageEnd2EndCommunicateController::setRequestReady(
        int sender, int receiver, std::string &&key) noexcept {
    pthread_rwlock_wrlock(&rwLock_);
    bothReadyRequests_.emplace(sender, receiver, key);
    pthread_rwlock_unlock(&rwLock_);
}

void SimpleMessageEnd2EndCommunicateController::messageSenderMain_() {
    using namespace std;
    bool shutdown = false;
    while (!shutdown) {
        queue<SharedMessage> msgs;
        waitForMessages_(
                &messagesToSendMutex_, &newMessageToSendCond_,
                senderReady_,
                messagesToSend_, msgs
        );

        while (!msgs.empty()) {
            auto msg(std::move(msgs.front()));
            msgs.pop();
            if (msg->type() == Message::DONE) {
                shutdown = true;
            }
            communicationImpl_->sendMessage(*msg);
        }

    }
}

void SimpleMessageEnd2EndCommunicateController::messageReceiverMain_() {
    using namespace std;
    bool shutdown = false;
    int rank = backend_->processRank();
    while (!shutdown) {
        setReady_(receiverReady_);

        auto msg(communicationImpl_->listen());

        switch (msg->type()) {
            case Message::READY: {
                if (msg->sender() == rank) {
                    // 消息的发送者是本进程, 那么表示对方也已准备好, 随时可以进行Tensor的通信
                    setRequestReady(msg->sender(), msg->receiver(), std::move(msg->movingMessage()));
                } else if (msg->receiver() == rank) {
                    // 消息的接收者是本进程, 那么发送者方肯定已经准备好
                    // 需要等待本进程的Tensor准备完毕, 再发送这个Message
                    // 给发送者进程, 表示这个Tensor已经准备好
                    RequestIdentifier id = make_tuple(
                            msg->sender(), msg->receiver(),
                            msg->message()
                    );
                    if (requestReady_(id)) {
                        // 告诉发送者进程本进程已经准备好
                        newMessageForwarding_(
                                &messagesToSendMutex_, &newMessageToSendCond_,
                                messagesToSend_, msg
                        );
                        // 注册已经准备好的Tensor
                        setRequestReady(msg->sender(), msg->receiver(), std::move(msg->movingMessage()));
                    } else {
                        waitingRequests_.emplace(std::move(id));
                    }
                } else SMCC_RECEIVING_UNEXPECTED_MESSAGE(msg, rank)
                break;
            }
            case Message::COMMUNICATE: {
                if (msg->sender() == rank) {
                    // 作为sender收到COMMUNICATE信息表示对方正在等待你发送数据,
                    // todo: last commit here
                } else if (msg->receiver() == rank) {
                    
                } else SMCC_RECEIVING_UNEXPECTED_MESSAGE(msg, rank)
            }
            case Message::DONE: {
                shutdown = true;
                break;
            }
        }
    }
}

void SimpleMessageEnd2EndCommunicateController::communicatorMain_() {
    using namespace std;
    bool shutdown = false;
    while (!shutdown) {

        queue<SharedMessage> msgs;
        waitForMessages_(
                &communicateMutex_, &newCommunicateTaskCond_,
                communicatorReady_,
                communicateMessages_, msgs
        );

        while (!msgs.empty()) {
            auto msg(std::move(msgs.front()));
            msgs.pop();
            if (msg->type() == Message::DONE) {
                shutdown = true;
            }
        }

    }
}

void SimpleMessageEnd2EndCommunicateController::newMessageForwarding_(
        pthread_mutex_t *handlingQueueMutex,
        pthread_cond_t *handlingCond,
        std::queue<SharedMessage> &handlingQueue,
        SharedMessage msg) {
    pthread_mutex_lock(handlingQueueMutex);
    handlingQueue.emplace(msg);
    // 因为这个条件变量只有一个线程等待, 所以可以使用signal, 若有多个线程等待, 则需要使用broadcast
    pthread_cond_signal(handlingCond);
    pthread_mutex_unlock(handlingQueueMutex);
}

void SimpleMessageEnd2EndCommunicateController::newMessageForwarding_(
        pthread_mutex_t *handlingQueueMutex,
        pthread_cond_t *handlingCond,
        std::queue<SharedMessage> &handlingQueue,
        std::queue<SharedMessage> &msgs) {
    pthread_mutex_lock(handlingQueueMutex);
    while (!msgs.empty()) {
        handlingQueue.emplace(std::move(msgs.front()));
        msgs.pop();
    }
    // 因为这个条件变量只有一个线程等待, 所以可以使用signal, 若有多个线程等待, 则需要使用broadcast
    pthread_cond_signal(handlingCond);
    pthread_mutex_unlock(handlingQueueMutex);
}


void SimpleMessageEnd2EndCommunicateController::waitForMessages_(
        pthread_mutex_t *queueMutex,
        pthread_cond_t *waitCond,
        bool &readyFlag,
        std::queue<SharedMessage> &queue,
        std::queue<SharedMessage> &result
) {
    pthread_mutex_lock(queueMutex);
    // 初始化工作: 此时线程已经准备就绪, 如果readyFlag不为true, 则置为true
    setReady_(readyFlag);

    while (queue.empty()) {
        pthread_cond_wait(waitCond, queueMutex);
    }

    // 从临界区中取出要发送的信息
    while (!queue.empty()) {
        result.emplace(std::move(queue.front()));
        queue.pop();
    }
    pthread_mutex_unlock(queueMutex);
}

SimpleMessageEnd2EndCommunicateController::~SimpleMessageEnd2EndCommunicateController() {
    sender_.join();
    receiver_.join();
    communicator_.join();
    pthread_rwlock_destroy(&rwLock_);
    pthread_mutex_destroy(&messagesToSendMutex_);
    pthread_cond_destroy(&newMessageToSendCond_);
}
}}}}