//
// Created by LYL232 on 2021/2/12.
//
#include <mutex>
#include "global/Global.h"
#include "global/initialize.h"
#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/collective/controller/rtc/RingTokenCommunicateController.h"
#include "communicate/collective/controller/rtc/mpi/MPIRingTokenCommunication.h"
#include "communicate/end2end/controller/bcc/BlockedEnd2EndCommunicateController.h"
#include "communicate/end2end/controller/bcc/mpi/MPIBlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl {

namespace initialize_implement {
std::recursive_mutex mutex_;
std::shared_ptr<MPIBackend> mpiBackend_;
std::shared_ptr<TensorsCollectiveCommunicateController> collectiveCommunicateController_;
std::shared_ptr<TensorEnd2EndCommunicateController> end2EndCommunicateController_;
std::shared_ptr<GlobalLog> logStream_;
}


std::shared_ptr<GlobalLog> globalLogGetter() {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!logStream_.get()) {
        string logFile("log-");
        int rank = communicationBackendGetter()->processRank();
        logFile.append(std::to_string(rank)).append(".txt");
        ofstream *log = new ofstream(logFile);
        function<void()> logStreamDestructor = [log]() {
            log->close();
            delete log;
        };
        logStream_.reset(new GlobalLog(*log, logStreamDestructor));
    }
    return logStream_;
}

std::shared_ptr<CommunicationBackend> communicationBackendGetter() {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!mpiBackend_.get()) {
        mpiBackend_.reset(new MPIBackend());
    }
    return mpiBackend_;
}

std::shared_ptr<TensorsCollectiveCommunicateController>
collectiveCommunicateControllerGetter() {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!collectiveCommunicateController_.get()) {
        auto backend = communicationBackendGetter();

        GLOBAL_INFO_WITH_THREAD_ID("new RingTokenCommunicateController")

        collectiveCommunicateController_.reset(
                new rtc::RingTokenCommunicateController(
                        backend,
                        make_shared<rtc::MPIRingTokenCommunication>(mpiBackend_)
                ));
        GLOBAL_INFO_WITH_THREAD_ID("new RingTokenCommunicateController initialized")
    }
    return collectiveCommunicateController_;
}

std::shared_ptr<TensorEnd2EndCommunicateController>
end2EndCommunicateControllerGetter() {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!end2EndCommunicateController_.get()) {
        auto backend = communicationBackendGetter();

        GLOBAL_INFO_WITH_THREAD_ID("new BlockedEnd2EndCommunicateController")

        end2EndCommunicateController_.reset(
                new bcc::BlockedEnd2EndCommunicateController(
                        backend,
                        make_shared<bcc::MPIBlockedEnd2EndCommunication>(mpiBackend_)
                ));
        GLOBAL_INFO_WITH_THREAD_ID("new BlockedEnd2EndCommunicateController initialized")
    }
    return end2EndCommunicateController_;
}

}}}