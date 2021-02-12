//
// Created by LYL232 on 2021/2/12.
//
#include <mutex>
#include "global/initialize.h"
#include "communicate/communication/mpi/MPIBackend.h"
#include "communicate/tensor/allreduce/rta/RingTokenAllreduceController.h"
#include "communicate/tensor/allreduce/rta/MPIRingTokenAllreduceCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl {

namespace initialize_implement {
std::recursive_mutex mutex_;
std::shared_ptr<MPIBackend> mpiBackend_;
std::shared_ptr<tensorsallreduce::TensorsAllreduceController> allreduceController_;
std::shared_ptr<GlobalLogStream> logStream_;
}


std::shared_ptr<GlobalLogStream> globalLogStreamGetter() {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!logStream_.get()) {
        string logFile("log-");
        int rank = communicationBackendGetter()->processRank();
        logFile.append(std::to_string(rank)).append(".txt");
        auto log = make_shared<ofstream>(logFile);
        function<void()> logStreamDestructor = [log]() {
            log->close();
        };
        logStream_.reset(new GlobalLogStream(log, logStreamDestructor));
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

std::shared_ptr<tensorsallreduce::TensorsAllreduceController> allreduceControllerGetter() {
    using namespace initialize_implement;
    using namespace std;
    lock_guard<recursive_mutex> guard(mutex_);
    if (!allreduceController_.get()) {
        allreduceController_.reset(
                new tensorsallreduce::rta::RingTokenAllreduceController(
                globalLogStreamGetter()->stream(),
                communicationBackendGetter(),
                make_shared<tensorsallreduce::rta::MPIRingTokenAllreduceCommunication>(mpiBackend_)
        ));
    }
    return allreduceController_;
}

}}}