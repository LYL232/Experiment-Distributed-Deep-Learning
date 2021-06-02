//
// Created by LYL232 on 2021/2/12.
//
#include <mutex>
#include "global/Global.h"
#include "global/initialize.h"
#include "communicate/tensor/collective/controller/rtc/mpi/MPIRingTokenCommunicateController.h"
#include "communicate/tensor/end2end/controller/bcc/BlockedEnd2EndCommunicateController.h"
#include "communicate/tensor/end2end/controller/bcc/mpi/MPIBlockedEnd2EndCommunication.h"
#include "communicate/message/mpi/MPIMessageController.h"

namespace lyl232 { namespace experiment { namespace ddl {

namespace initialize_implement {
std::recursive_mutex mutex_;
std::shared_ptr<MPIBackend> mpiBackend_;
std::shared_ptr<TensorsCollectiveCommunicateController> collectiveCommunicateController_;
std::shared_ptr<TensorEnd2EndCommunicateController> end2EndCommunicateController_;
std::shared_ptr<GlobalLog> logStream_;
std::shared_ptr<MPIMessageController> mpiMessageController_;
std::shared_ptr<HeapMemoryManager> heapMemoryManager_;
}


std::shared_ptr<GlobalLog> globalLogGetter() noexcept {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!logStream_.get()) {
        string logFile("log-");
        int rank = communicationBackendGetter()->worldCommunicator()->rank();
        logFile.append(std::to_string(rank)).append(".txt");
        auto *log = new ofstream(logFile);  // no mem track
        function<void()> logStreamDestructor = [log]() {
            log->close();
            delete log;  // no mem track
        };
        logStream_.reset(new GlobalLog(*log, logStreamDestructor));   // no mem track
    }
    return logStream_;
}

std::shared_ptr<HeapMemoryManager> heapMemoryManagerGetter() noexcept {
    using namespace initialize_implement;
    using namespace std;
    lock_guard <recursive_mutex> guard(mutex_);
    if (!heapMemoryManager_.get()) {
        heapMemoryManager_.reset(new HeapMemoryManager());  // no mem track
    }
    return heapMemoryManager_;
}

std::shared_ptr<CommunicationBackend> communicationBackendGetter() noexcept {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!mpiBackend_.get()) {
        mpiBackend_.reset(new MPIBackend());  // no mem track
    }
    return mpiBackend_;
}

std::shared_ptr<TensorsCollectiveCommunicateController>
collectiveCommunicateControllerGetter() noexcept {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!collectiveCommunicateController_.get()) {
        auto backend = communicationBackendGetter();
        GLOBAL_INFO_WITH_THREAD_ID("new MPIRingTokenCommunicateController")
        collectiveCommunicateController_.reset(new rtc::MPIRingTokenCommunicateController());  // no mem track
        GLOBAL_INFO_WITH_THREAD_ID("new MPIRingTokenCommunicateController initialized")
    }
    return collectiveCommunicateController_;
}

std::shared_ptr<TensorEnd2EndCommunicateController>
end2EndCommunicateControllerGetter() noexcept {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!end2EndCommunicateController_.get()) {
        auto backend = communicationBackendGetter();

        GLOBAL_INFO_WITH_THREAD_ID("new BlockedEnd2EndCommunicateController")

        end2EndCommunicateController_.reset(
                new bcc::BlockedEnd2EndCommunicateController(  // no mem track
                        backend,
                        make_shared<bcc::MPIBlockedEnd2EndCommunication>(mpiBackend_)
                ));
        GLOBAL_INFO_WITH_THREAD_ID("new BlockedEnd2EndCommunicateController initialized")
    }
    return end2EndCommunicateController_;
}

std::shared_ptr<MessageController> messageControllerGetter() noexcept {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!mpiMessageController_.get()) {
        auto backend = communicationBackendGetter();
        GLOBAL_INFO_WITH_THREAD_ID("new MPIMessageController")
        mpiMessageController_.reset(new MPIMessageController(mpiBackend_));  // no mem track
        GLOBAL_INFO_WITH_THREAD_ID("new MPIMessageController initialized")
    }
    return mpiMessageController_;
}

}}}